from typing import Dict, Tuple, Union

import pandas as pd

DfOrSeries = Union[pd.DataFrame, pd.Series]
PdDTypeQuadTuple = Tuple[DfOrSeries, DfOrSeries, DfOrSeries, DfOrSeries]
StrDict = Dict[str, str]
