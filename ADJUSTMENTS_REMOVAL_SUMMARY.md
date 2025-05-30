# Summary of Adjustments Removal

## Date: January 30, 2025

### What was removed:

1. **Backend (src/controllers/views.py)**:
   - Removed `/api/adjustments/list` endpoint
   - Removed `/api/adjustments/delete/<int:adj_id>` endpoint
   - Removed `adjustments` parameter from forecast request processing

2. **Frontend (templates/index.html)**:
   - Removed "Добавить корректировку" button
   - Removed "Менеджер корректировок" button
   - Removed adjustments list div
   - Removed JavaScript variable `let adjustments = []`
   - Removed functions:
     - `renderAdjustments()`
     - `removeAdjustment(index)`
     - `loadAdjustmentManager()`
     - `deleteAdjustmentFromDB(id)`
   - Removed event handlers:
     - `$('#addAdjustment').click()`
     - `$('#saveAdjustment').click()`
     - `$('#viewAdjustments').click()`
   - Removed adjustments parameter from forecast API call
   - Removed all adjustment-related CSS classes

3. **CSS styles removed**:
   - `.adjustment-panel`
   - `.adjustment-item`
   - `.adjustment-item:hover`

### Current state:
- The application now focuses on the Settings tab for per-cafe configuration
- All adjustment-related functionality has been cleanly removed
- The Settings tab remains functional with seasonality and model sensitivity settings per cafe
- The forecasting functionality continues to work without adjustments

### Testing required:
1. Run the application and verify that:
   - No adjustment-related UI elements are visible
   - Settings tab works correctly
   - Forecast calculation works without errors
   - Passport tab displays cafe information correctly