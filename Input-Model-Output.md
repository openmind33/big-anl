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

http://memo.polypia.net/archives/2210
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

Files in Remote Server
<pre>
import subprocess
ssh = subprocess.Popen(['ssh', 'user@host', 'cat', 'path/to/file'],
                       stdout=subprocess.PIPE)
for line in ssh.stdout:
    line  # do stuff


import subprocess

subprocess.Popen(["rsync", host-ip+'/path/to/file'],stdout=subprocess.PIPE)
for line in ssh.stdout:
    line  # do stuff


import os
import paramiko

ssh = paramiko.SSHClient() 
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
sftp.put(localpath, remotepath)
sftp.close()
ssh.close()
</pre>

HDFS

[reference]

- http://wesmckinney.com/blog/python-hdfs-interfaces/
<pre>
from pyarrow import HdfsClient

# Using libhdfs
hdfs = HdfsClient(host, port, username, driver='libhdfs')

# Using libhdfs3
hdfs_alt = HdfsClient(host, port, username, driver='libhdfs3')

with hdfs.open('/path/to/file') as f:
    ...
    
    
from hdfs3 import HDFileSystem

hdfs = HDFileSystem(host, port, user)
with hdfs.open('/path/to/file', 'rb') as f:
    ...
</pre>

hadoop-spark-sklearn :  http://www.sunlab.org/teaching/cse8803/fall2016/lab/spark-mllib/


https://stackoverflow.com/questions/33791535/how-to-save-numpy-array-from-pyspark-worker-to-hdfs-or-shared-file-system
<pre>
imports:
from hdfs import InsecureClient
from tempfile import TemporaryFile
create a hdfs client. In most cases, it is better to have a utility function somewhere in your script, like this one:
def get_hdfs_client():
    return InsecureClient("<your webhdfs uri>", user="<hdfs user>",
         root="<hdfs base path>")
load and save your numpy inside a worker function:
hdfs_client = get_hdfs_client()

# load from file.npy
path = "/whatever/hdfs/file.npy"
tf = TemporaryFile()

with hdfs_client.read(path) as reader:
    tf.write(reader.read())
    tf.seek(0) # important, set cursor to beginning of file

np_array = numpy.load(tf)

----start of pseudo code by 이이백---
model = DecisionTreeClassifier()
  .fit(np_array[:,[3,4,5]]   # Feature Selection
      ,np_array[:,[1]]       # Label Assignment

----end of pseudo code by 이이백---



# save to file.npy
tf = TemporaryFile()
numpy.save(tf, np_array)
tf.seek(0) # important ! set the cursor to the beginning of the file
# with overwrite=False, an exception is thrown if the file already exists
hdfs_client.write("/whatever/output/file.npy", tf.read(),  overwrite=True) 
Notes: 
the uri used to create the hdfs client begins with http://, because it uses the web interface of the hdfs file system;
ensure that the user you pass to the hdfs client has read and write permissions
in my experience, the overhead is not significant (at least in term of execution time)
the advantage of using tempfiles (vs regular files in /tmp) is that you ensure no garbage files stay in the cluster machines after the script ends, normally or not
</pre>



