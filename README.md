<h1>빅데이터 분석 전문가 과정</h1>
<h2>탐색과 기술통계</h2>
<h3>탐색</h3>

<h4>using SQL in Pandas</h4>
<pre>
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
pysqldf("select * from dfTrain limit 10;")
pysqldf("select Y1, sum(*) as sum from dfTrain group by Y2;")
</pre>



<h4>Histogram</h4>
<pre>
df_hist = dfTrain[['Y1']].hist(bins=10)
df_hist = dfTrain[['Y2']].hist(bins=2)
</pre>



<h3>기술통계</h3>

<h3>연속형 변수의 기술통계량</h3>
<pre>
dfTrain.describe()
</pre>

<h3>연속형 변수의 기술통계량 - 분포의 모양 - 왜도, 첨도</h3>
<pre>
for col in dfTrain.columns:
    (kurtosis,skew)=dfTrain[col].kurtosis(),dfTrain[col].skew()
    print("{}: 첨도={}, 왜도={}".format(col,kurtosis,skew))
</pre>
<pre>
Y1: 첨도=0.0113768715583209, 왜도=-0.18034727886048718
Y2: 첨도=-1.0287224064997245, 왜도=0.9856385635666307
V1: 첨도=-1.5056814870136943, 왜도=-0.175505573216797
V2: 첨도=-1.2919483985912386, 왜도=-0.11837633149409198
V3: 첨도=-1.186659656950666, 왜도=0.37866074546825
V4: 첨도=-1.0950732744326248, 왜도=0.019066354744059412
V5: 첨도=-0.6730584014647358, 왜도=-0.04065900020327655
V6: 첨도=-0.8521450426812813, 왜도=-0.7651623109781867
V7: 첨도=0.1668594015765823, 왜도=0.9077316290704271
V8: 첨도=-1.6686448659451378, 왜도=-0.011235555220714177
V9: 첨도=-0.15423533517418653, 왜도=1.1488567009424828
V10: 첨도=0.016457144971524507, 왜도=1.1211187262055549
V11: 첨도=-0.46891067624666727, 왜도=-0.20318697986064396
</pre>

<h2>Decision Tree - Classification & Regression</h2>

- conda install graphviz
- pip install graphviz
- download and install : graphviz-2.38.msi
    - [download link] https://graphviz.gitlab.io/_pages/Download/Download_windows.html
    - [set path] C:\Apps\Graphviz2.38\bin



<h2>Naive Bayes - Classification</h2>




<h2>Naive Bayes</h2>
- http://chem-eng.utoronto.ca/~datamining/dmc/naive_bayesian.htm

<h2>Model Evaluation</h2>
- https://en.wikipedia.org/wiki/Receiver_operating_characteristic


<h2>DTW</h2>
- https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html

<h2>Word2Vector using Gensim</h2>
- pip install gensim
$ ./gensim-word2vec.py



