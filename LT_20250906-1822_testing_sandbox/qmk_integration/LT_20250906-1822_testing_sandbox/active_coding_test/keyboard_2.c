
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

// Function: process_keyboard_matrix
// Purpose: Handle keyboard matrix scanning
int process_keyboard_matrix(void) {
    uint8_t matrix_state[MATRIX_ROWS];
    
    for (int row = 0; row < MATRIX_ROWS; row++) {
        matrix_state[row] = scan_row(row);
        
        if (matrix_state[row] != previous_state[row]) {
            handle_key_change(row, matrix_state[row]);
        }
    }
    
    return 0;
}
            
// Generated: 2025-09-06T22:21:05.268272
// Test ID: ffad578f-7c16-44c8-b380-839608ca275f

// SYNC_TEST_MARKER: sync-test-1757190065268329