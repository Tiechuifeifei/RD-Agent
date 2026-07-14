import qlib
from qlib.data import D

QLIB_DATA = "/Users/yufei/wrds_project/staging/qlib_data"
START_TIME = "2008-01-01"
END_TIME = "2023-12-31"
FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]


def main() -> None:
    qlib.init(provider_uri=QLIB_DATA, region="us")
    instruments = D.instruments("sp500")

    data = (
        D.features(instruments, FIELDS, start_time=START_TIME, end_time=END_TIME, freq="day")
        .swaplevel()
        .sort_index()
    )
    data.to_hdf("./daily_pv_all.h5", key="data")

    debug = D.features(
        instruments,
        FIELDS,
        start_time="2018-01-01",
        end_time="2019-12-31",
        freq="day",
    ).swaplevel().sort_index()

    top_instruments = debug.index.get_level_values("instrument").unique()[:100]
    debug = debug.loc[(slice(None), top_instruments), :].sort_index()
    debug.to_hdf("./daily_pv_debug.h5", key="data")


if __name__ == "__main__":
    main()
