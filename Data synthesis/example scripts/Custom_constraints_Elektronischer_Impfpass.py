import pandas as pd
from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):
    vaccination = column_names[0]
    vaccine = column_names[1]
    manufacturer = column_names[2]
    vaccination_date = column_names[3]
    expiry_date = column_names[4]

    return (
        (((~data[vaccination]) & (data[vaccine].isnull()) &
          (data[manufacturer].isnull()) & (data[vaccination_date].isnull()) &
          (data[expiry_date].isnull())) |
         ((data[vaccination]) & (~ data[vaccine].isnull()) &
          (~ data[manufacturer].isnull()) & (~ data[vaccination_date].isnull()) &
          (~ data[expiry_date].isnull()) & (data[vaccination_date] < data[expiry_date])))
    )


Custom_Constraint_Elektronischer_Impfpass = create_custom_constraint_class(
    is_valid_fn=is_valid,
    transform_fn=None,
    reverse_transform_fn=None
)
