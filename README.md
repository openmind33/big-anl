<h1>빅데이터 분석 전문가 과정</h1>
<h2>탐색과 기술통계</h2>
<h4>SQL in Pandas</h4>
<pre>
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
pysqldf("select * from dfTrain limit 10;")

pysqldf("select Y1 from dfTrain group by Y2;")
</pre>



<h2>Naive Bayes</h2>
- http://chem-eng.utoronto.ca/~datamining/dmc/naive_bayesian.htm

