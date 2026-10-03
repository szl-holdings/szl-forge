"""Admit the pinned CPU wheel payload before importing any installed package.

Run with the admitted native Python image and -I -S -B. The explicitly admitted
site directory is activated without executing .pth files or global site hooks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import zipfile

LOCK_SHA256 = '952e153b19c8d11875ddd652660cf9b449b23bff87845e7a6aae978bd91bff49'
LOCK = Path(__file__).with_name('cpu-runtime-lock.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest_stream(handle):
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def digest_file(path):
    with Path(path).open('rb') as handle:
        return digest_stream(handle)


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and not any(ord(c) < 32 for c in name), 'Unsafe wheel member name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and all(part not in ('.', '..') for part in path.parts), 'Unsafe wheel member path')
    reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}
    require(all(part.rstrip(' .') == part and part.split('.')[0].casefold() not in reserved for part in path.parts), 'Unsafe Windows wheel member')
    return path


def no_links(path, recursive=False):
    path = Path(path).absolute()
    cursor = path
    while cursor != cursor.parent:
        if cursor.exists() or cursor.is_symlink():
            info = cursor.lstat()
            require(not cursor.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Links/reparse points are not admitted')
        cursor = cursor.parent
    if recursive:
        for child in path.rglob('*'):
            info = child.lstat()
            require(not child.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Environment contains a link/reparse point')
    return path.resolve(strict=True)


def profile_path(path, directory=True):
    result = no_links(path)
    profile = no_links(Path.home())
    require(result != profile and result.is_relative_to(profile) and result.anchor.casefold() == profile.anchor.casefold(), 'Runtime paths must remain below the current-user profile')
    require(result.is_dir() if directory else result.is_file(), 'Runtime path has the wrong type')
    return result


def native_image():
    require(os.name == 'nt', 'The managed CPU runtime requires Windows')
    buffer = ctypes.create_unicode_buffer(32768)
    require(ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer)) != 0, 'Native Python image is unavailable')
    return profile_path(buffer.value, directory=False)


def read_lock():
    require(digest_file(LOCK) == LOCK_SHA256, 'CPU wheel lock is not the canonical admitted source')
    lock = json.loads(LOCK.read_bytes())
    require(lock['schema'] == 'szl.foundation-confirmation.cpu-wheel-lock/v1' and lock['python'] == '3.11' and lock['platform'] == 'win_amd64' and lock['torch_version'] == '2.10.0+cpu' and len(lock['wheels']) == 11, 'CPU wheel contract differs')
    return lock


def verify_payload(environment, wheelhouse, records):
    """Check authoritative wheel bytes and their installed importable payload."""
    environment = no_links(environment, recursive=True)
    wheelhouse = no_links(wheelhouse, recursive=True)
    site = environment / 'Lib/site-packages'
    require(site.is_dir(), 'The isolated CPU site-packages directory is missing')
    expected = {}
    expected_folded = set()
    allowed_metadata = set()
    wheel_names = set()
    package_versions = {}
    for row in records:
        require(re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None, 'Wheel hash is incomplete')
        filename = row['filename']
        require(len(safe_name(filename).parts) == 1 and filename not in wheel_names, 'Wheel filename is unsafe or duplicated')
        wheel_names.add(filename)
        wheel = wheelhouse / filename
        require(wheel.is_file() and wheel.stat().st_size == row['download_bytes'] and digest_file(wheel) == row['sha256'], 'Official wheel bytes differ: ' + row['name'])
        package_versions[row['name']] = row['version']
        with zipfile.ZipFile(wheel) as archive:
            require(sum(info.file_size for info in archive.infolist()) == row['expanded_bytes'], 'Wheel expanded-size contract differs')
            case_names = set()
            for info in archive.infolist():
                # ZipInfo normalizes Windows backslashes and truncates NULs;
                # inspect its original archive name before trusting filename.
                safe_name(info.orig_filename.rstrip('/') if info.is_dir() else info.orig_filename)
                require(info.orig_filename == info.filename, 'Unsafe normalized wheel member name')
                name = info.filename.rstrip('/') if info.is_dir() else info.filename
                safe_name(name)
                require(name.casefold() not in case_names, 'Duplicate case-folded wheel member')
                case_names.add(name.casefold())
                if info.is_dir():
                    continue
                require(not (info.external_attr >> 16) & 0o170000 == 0o120000, 'Wheel symlink is not admitted')
                parts = PurePosixPath(name).parts
                if parts[0].endswith('.data'):
                    require(len(parts) >= 3, 'Invalid wheel data mapping')
                    if parts[1] not in ('purelib', 'platlib'):
                        continue  # Scripts/headers/data are not on the import path.
                    name = '/'.join(parts[2:])
                if parts[0].endswith('.dist-info'):
                    allowed_metadata.update(parts[0] + '/' + field for field in ('RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json', 'uv_cache.json'))
                if name in allowed_metadata:
                    continue  # Installer-only metadata never authorizes source.
                require(name.casefold() not in expected_folded, 'Installed payload collision')
                expected_folded.add(name.casefold())
                # Stream large DLLs instead of allocating their expanded bytes.
                with archive.open(info) as payload_file:
                    expected[name] = digest_stream(payload_file)
    installed = []
    for path in site.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(site).as_posix()
        require(relative in expected or relative in allowed_metadata, 'Unexpected importable environment file: ' + relative)
        if relative.endswith('/REQUESTED'):
            require(path.read_bytes() == b'', 'Installer REQUESTED marker differs')
        if relative.endswith('/INSTALLER'):
            require(path.read_bytes() == b'uv', 'Installer identity differs')
        if relative in expected:
            installed.append((relative, path))
    # Validate every pathname before opening payloads. Four bounded I/O workers
    # avoid serial file-open delay; every admitted file still gets a full hash.
    def verify_installed(entry):
        relative, path = entry
        observed_hash = digest_file(path)
        require(observed_hash == expected[relative], 'Installed wheel payload differs: ' + relative)
        return relative, observed_hash
    with ThreadPoolExecutor(max_workers=4) as workers:
        observed = dict(workers.map(verify_installed, installed))
    require(observed == expected, 'Installed CPU wheel payload is incomplete')
    return {'site_packages': str(site), 'environment_payload_sha256': hashlib.sha256(canonical(observed)).hexdigest(), 'files_verified': len(observed), 'package_versions': package_versions}


def admit(environment, wheelhouse, expected_python_sha256):
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode, 'Use the admitted Python image with -I -S -B')
    require(sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8, 'The pinned CPU wheel requires Python3.11 AMD64')
    executable = native_image()
    require(re.fullmatch(r'[0-9a-f]{64}', expected_python_sha256) is not None and digest_file(executable) == expected_python_sha256, 'Native Python image hash differs')
    environment = profile_path(environment)
    wheelhouse = profile_path(wheelhouse)
    config = environment / 'pyvenv.cfg'
    require(config.is_file(), 'The owned environment has no pyvenv.cfg')
    settings = {line.partition('=')[0].strip(): line.partition('=')[2].strip() for line in config.read_text().splitlines() if '=' in line}
    require(settings.get('include-system-site-packages') == 'false' and Path(settings.get('home', '')).resolve() == executable.parent, 'Environment base or global-site isolation differs')
    lock = read_lock()
    payload = verify_payload(environment, wheelhouse, lock['wheels'])
    binding = {'native_executable': str(executable), 'python_image_sha256': expected_python_sha256, 'python_version': sys.version, 'environment_root': str(environment), 'site_packages': payload['site_packages'], 'wheelhouse': str(wheelhouse), 'lock_sha256': LOCK_SHA256, 'environment_payload_sha256': payload['environment_payload_sha256'], 'package_versions': payload['package_versions']}
    return {'schema': 'szl.foundation-confirmation.cpu-environment/v1', 'state': 'VERIFIED', 'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'binding': binding, 'binding_sha256': hashlib.sha256(canonical(binding)).hexdigest(), 'files_verified': payload['files_verified'], 'packages_verified': len(lock['wheels']), 'global_site_packages_loaded': False, 'pth_files_executed': False, 'unsigned': True}


def activate(admission):
    require(admission['state'] == 'VERIFIED', 'CPU environment was not admitted')
    require(not any('site-packages' in value.casefold() for value in sys.path), 'A site directory was already active')
    sys.path.insert(0, admission['binding']['site_packages'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-root', required=True)
    parser.add_argument('--wheelhouse', required=True)
    parser.add_argument('--expected-python-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.environment_root, args.wheelhouse, args.expected_python_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
