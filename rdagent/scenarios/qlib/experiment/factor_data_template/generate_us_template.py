import qlib
from qlib.data import D

def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")

    instruments = D.instruments("nasdaq100")
    fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

    data = D.features(
        instruments,
        fields,
        start_time="2009-01-01",
        end_time="2020-11-06",
        freq="day",
    ).swaplevel().sort_index()

    data.to_hdf("./daily_pv_all.h5", key="data")

    debug = D.features(
        instruments,
        fields,
        start_time="2018-01-01",
        end_time="2019-12-31",
        freq="day",
    ).swaplevel().sort_index()

    debug.to_hdf("./daily_pv_debug.h5", key="data")

if __name__ == "__main__":
    main()
