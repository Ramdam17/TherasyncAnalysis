# MOI Annotations - Known Issues

## Session Duration Inconsistency

**Issue**: Timestamp discrepancy between MOI annotations and session metadata.

### Example: G01/Session 01
- **MOI annotations max timestamp**: 01:05:23 (3923s = 65.4 minutes)
- **Session duration (metadata)**: 2779s (46.3 minutes)
- **Discrepancy**: ~19 minutes

### Potential Causes
1. Session metadata duration based on physiological recording (participant 01 BVP)
2. Physiological recording may have stopped before end of therapy session
3. MOI annotations come from video recording which may be longer
4. Possible data collection issues (equipment stopped early)

### Impact
- Epoching currently uses metadata duration (46.3 min)
- Some MOI annotations extend beyond epoch range
- Visualizations and statistics may not cover full session

### Action Required
- ✅ Document issue (this file)
- ⏳ Check with colleague about correct session durations
- ⏳ Verify video recording lengths vs physiological recording lengths
- ⏳ Decide on authoritative source for session duration (video vs physio)
- ⏳ Potentially regenerate MOI sidecars with corrected durations

### Status
**PENDING** - Awaiting colleague feedback on correct session durations.

---
*Last updated: 2025-11-18*
*Contact: Lena Adel / Remy Ramadour*
