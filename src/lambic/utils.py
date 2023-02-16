def filter_xarray(psd_concat, coordinate : str, contains):
    def filter_substring(string, substr):
        return [str for str in string if
                any(sub in str for sub in substr)]
    # Driver code
    roi_values = psd_concat.coords[coordinate].values.tolist()
    print(filter_substring(roi_values, contains))
    filtered_roi = filter_substring(roi_values, contains)
    roi_names_set = psd_concat.sel(roi_names=filtered_roi)
    return roi_names_set