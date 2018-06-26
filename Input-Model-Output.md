<h2></h2>

[Source]
DBMS
- DB의 Table-> SQL -> Numpy Array로 변환 -> Analytic Model에 적용
  - db = dbms.connect.postgres('UserName', 'SuperSecret', 'Chinook')
  - 파이선 오라클 연동 : http://digndig.net/?p=348
<pre>
import cx_Oracle
dsn = cx_Oracle.makedsn("HOST", PORT_NUMBER, "DB_NAME")
db = cx_Oracle.connect("ID", "PASSWORD", dsn)
cursor = db.cursor()
 
cursor.execute("""SELECT * FROM TABLE_NAME""")
row = cursor.fetchone()
</pre>

<pre> 
import cx_Oracle
 
PORT_NUM = 1521
dsn = cx_Oracle.makedsn("SERVER_HOST", PORT_NUM, "ORCLE_SID_NAME")
db = cx_Oracle.connect("USERNAME", "PASSWORD", dsn)
cursor = db.cursor()
 
cursor.execute("""SELECT * FROM sso_data.t_users where rownum < 100""")
row = cursor.fetchone()
while row:
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " + str(row[3]))
    row = cursor.fetchone()
</pre>

File 
HDFS


