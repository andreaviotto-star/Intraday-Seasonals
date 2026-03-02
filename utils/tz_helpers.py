from datetime import datetime
import pytz

def localize_et_index(idx):
    return idx.tz_localize("US/Eastern")

def convert_to_zurich(idx):
    et = pytz.timezone("US/Eastern")
    zur = pytz.timezone("Europe/Zurich")
    return idx.tz_convert(zur)
