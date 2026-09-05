# LIVE_SHADOW binding-directory example

This directory contains a deidentified example only. It is not a patient
identity registry, a deidentification service, a production mapping, or
authorization to use clinical data.

The example JSON filename is SHA-256 over the UTF-8 bytes of PID-3, one NUL
byte, and OBR-2. If OBR-2 is empty, the bridge uses OBR-3. The included file
corresponds only to the placeholder source tokens DEID-SUBJECT01 and
DEID-ORDER0001.

A site integration must provision each binding outside the browser, validate
the authoritative patient/order/specimen relationship, restrict filesystem
access, and retain the exact source-to-pseudonym mapping under approved site
policy. A hash filename is only an index and does not prove identity,
deidentification, authorization, or clinical correctness.
