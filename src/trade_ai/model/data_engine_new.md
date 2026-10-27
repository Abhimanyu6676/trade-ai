# data_engine_new class


create a new file `data_engine_new.py`
in this file, create a sample class that will hold a map of array of ohlcv candles in python, this class will be singleton and will share the same instance no matter how many and which other file import it, its data should survive and will pe provide. 

just like we do in js as

```js
const data = map<string, OhlcvCandle>
```
use `OhlcvCandle` from `openalgoTypes.py`

the map string would be `{ticker}:{exchange}` structure

the map will hold the candles and a getter and setter for them,

and it'll preserve the data such that if one file set a value and another later calls for data it'll get update data