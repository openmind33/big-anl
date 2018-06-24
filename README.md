<h1>빅데이터 분석 전문가 과정</h1>
<h2>탐색과 기술통계</h2>
<h3>탐색</h3>

<h4>using SQL in Pandas</h4>
<pre>
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
pysqldf("select * from dfTrain limit 10;")
pysqldf("select Y1 from dfTrain group by Y2;")
</pre>


<h3>기술통계</h3>

<h3>연속형 변수의 기술통계량</h3>
<pre>
dfTreain.describe()
</pre>


<h4>Histogram</h4>
<pre>
df_hist = dfTrain[['Y2']].hist(bins=2)
</pre>

<h3>기술통계량</h3>
<pre>
</pre>




<h4>기술통계량</h4>

<h4>SQL in Pandas</h4>

<h2>Decision Tree - Classification & Regression</h2>

<h2>Naive Bayes - Classification</h2>




<h2>Naive Bayes</h2>
- http://chem-eng.utoronto.ca/~datamining/dmc/naive_bayesian.htm

