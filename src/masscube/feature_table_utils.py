# Utility functions for feature table manipulation

def calculate_fill_percentage(feature_table, individual_sample_groups):
    """
    calculate fill percentage for each feature

    Parameters
    ----------
    feature_table : pd.DataFrame
        feature table
    individual_sample_groups : list
        list of individual sample groups

    Returns
    -------
    feature_table : pd.DataFrame
        feature table with fill percentage
    """

    # blank samples are not included in fill percentage calculation
    blank_number = len([x for x in individual_sample_groups if 'blank' in x])
    total_number = len(individual_sample_groups)
    if blank_number == 0:
        feature_table['fill_percentage'] = (feature_table.iloc[:, -total_number:] > 0).sum(axis=1)/total_number * 100
    else:
        feature_table['fill_percentage'] = (feature_table.iloc[:, -total_number:-blank_number] > 0).sum(axis=1)/(total_number - blank_number) * 100
    return feature_table