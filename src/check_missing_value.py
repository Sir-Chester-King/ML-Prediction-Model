import numpy


# ---- CHECK MISSING VALUE SECTION ---- #
# Check if there is/are missing values in the matrix_features_independent dataset.
def check_missing_value(matrix_features_independent):
    # Returns a boolean mask indicating the presence of NaN
    nan_mask = numpy.isnan(matrix_features_independent)

    # Returns True if there is at least one NaN value in the array, False otherwise.
    has_nan = numpy.any(nan_mask)

    if has_nan:
        return True
# ---- CHECK MISSING VALUE SECTION ---- #
