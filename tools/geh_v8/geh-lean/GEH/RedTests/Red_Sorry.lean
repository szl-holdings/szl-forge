import GEH.Compliance.GehGuard
-- RED TEST: must FAIL the build. sorry => sorryAx outside the trust base.
theorem geh_red_sorry (p : Prop) : p := by sorry
#geh_guard geh_red_sorry
