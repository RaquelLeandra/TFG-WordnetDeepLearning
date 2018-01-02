import matplotlib.pyplot as plt
import numpy as np

changes = np.matrix([['$C_{(-1,-1)}$', '$C_{(0,-1)}$', '$C_{(1,-1)}$'],
                    ['$C_{(-1,0)}$', '$C_{(0,0)}$', '$C_{(1,0)}$'],
                    ['$C_{(-1,1)}$', '$C_{(0,1)}$', '$C_{(1,1)}$']])
fig, ax = plt.subplots(figsize=(5, 5))
diag = np.zeros([3, 3])
diag[0, 0] += 1
diag[1, 1] += 1
diag[2, 2] += 1
ax.matshow(diag, cmap=plt.cm.Blues, alpha=0.3)
for i in range(changes.shape[0]):
	for j in range(changes.shape[1]):
		ax.text(x=j, y=i, s=changes[i, j], va='center', ha='center', fontsize=20)
plt.xticks([0, 1, 2], [-1, 0, 1])
plt.yticks([0, 1, 2], [-1, 0, 1])
plt.xlabel('Original values')
plt.ylabel('New values')
plt.title('Changes from synset1 to synset2')
plt.tight_layout()
#plt.show()
plt.savefig('Changes from synset1 to synset2')
