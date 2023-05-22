import pandas as pd
from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):
    return(((data['status'] == 'Died') & (data['date_recovered'].isnull())) | (
                (data['status'] != 'Died') & (data['date_of_death'].isnull())))


Custom_Constraint_Case_Information_Recovery_Death_Date = create_custom_constraint_class(
    is_valid_fn=is_valid,
    transform_fn=None,
    reverse_transform_fn=None
)
