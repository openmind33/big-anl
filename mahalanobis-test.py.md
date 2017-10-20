<pre>
"""
Mahalanobis - get Similarity Distance
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)

import mahalanobis

x = [1,2,3,4,5,5,100]
y = [1,2,3,4,1,1,1]
# Remove Outlier
dist1 = mahalanobis.MD_removeOutliers(x, y)
print('removed :', dist1)

</pre>
