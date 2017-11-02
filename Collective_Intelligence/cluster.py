from PIL import Image, ImageDraw
# 加载数据文件
def readfile(filename):
    lines = [line for line in open(filename)]
    # 第一行是列标题
    colnames = lines[0].strip().split('\t')[1:]
    rownames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split('\t')
        # 每行的第一列是行名
        rownames.append(p[0])
        # 剩余部分就是该行对应的数据
        data.append([float(x) for x in p[1:]])
        print("Readfile Done...\n")
    return rownames, colnames,data

from math import sqrt
def pearson(v1, v2):
    # 简单求和
    sum1 = sum(v1)
    sum2 = sum(v2)
    # 求平方和
    sum1Sq = sum([pow(v,2) for v in v1])
    sum2Sq = sum([pow(v,2) for v in v2])
    #求乘积之和
    pSum = sum([v1[i]*v2[i] for i in range(len(v1))])
    # 计算r(Pearson score)
    num = pSum-(sum1*sum2/len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2)/len(v1))*(sum2Sq - pow(sum2, 2)/len(v1)))
    if den == 0: return 0
    # 返回的是以1.0减去皮尔逊相关度之后的结果，目的是为了让相似度越大的两个元素之间的距离变得更小
    return 1.0-num/den

class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance

def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1
    # 最开始的聚类就是数据集中的行
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]

    while len(clust)>1:
        lowestpair = (0,1)
        closest = distance(clust[0].vec, clust[1].vec)
        # 遍历每一个配对， 寻找最小距离
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                # 用distance来缓存距离的计算值
                if(clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]

                # 更新最小距离及其对应的配对
                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        print("len(clust) now is "+str(len(clust)))
        # 计算两个聚类的平均值
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]
        # 建立新的聚类
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance = closest, id=currentclustid)

        # 不在原始集合中的聚类， 其id为负数
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]

blognames, words, data = readfile('./data/blogdata.txt')
clust=hcluster(data)

# 递归遍历聚类树，并将其以类似文件系统层级结构的形式打印出来
def printclust(clust, labels=None, n=0):
    # 利用缩进来建立层级布局
    for i in range(n) : print(' '),
    if clust.id < 0:
        # 负数标记代表这是一个分支
        print('-')
    else:
        # 正数标记代表这是一个叶节点
        if labels == None: print(clust.id)
        else: print(labels[clust.id])

    # 现在开始打印右侧分支和左侧分支
    if clust.left != None: printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None: printclust(clust.right, labels=labels, n=n+1)

printclust(clust, labels=blognames)

# 返回给定聚类的总体高度
def getheight(clust):
    # 这是一个叶节点吗？若是，则高度为1
    if clust.left == None and clust.right == None: return 1
    # 否则， 高度为每个分支的高度之和
    return getheight(clust.left)+getheight(clust.right)

# 得到误差深度，一个结点的误差深度等于其下所属的每个分支的最大可能误差
# 因为线条的长度会根据每个节点的误差进行相应的调整，所以我们需要根据总的误差值生成一个缩放因子(scaling factor)
def getdepth(clust):
    # 一个叶节点的距离是0.0
    if clust.left == None and clust.right == None: return 0
    # 一个枝结点的距离等于左右两侧分支中距离较大者
    # 加上该枝结点自身的距离
    return max(getdepth(clust.left), getdepth(clust.right))+clust.distance

# 函数drawdendrogram为每一个最终生成的聚类创建一个高度为20像素，宽度固定的图片。
# 其中的缩放因子是由固定宽度除以总的深度值得到的
# 该函数为图片建立相应的draw对象，然后在根结点的位置调用drawnode函数，并令其处于整幅图片左侧正中间的位置
def drawdendrogram(clust, lables, jpeg = 'clusters.jpg'):
    # 宽度和高度
    h = getheight(clust)*20
    w = 1200
    depth = getdepth(clust)

    # 由于宽度是固定的，因此我们需要对距离值做相应的调整
    scalling = float(w-150)/depth
    # 新建一个白色背景的图片
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h/2, 10, h/2), fill=(255,0,0))

    # 画第一个节点
    drawnode(draw, clust, 10, (h/2), scalling, lables)
    img.save(jpeg, 'JPEG')


# 接受一个聚类及其位置作为输入参数。函数取到子节点的高度，并计算出这些节点所在的位置，然后用线条将它们连起来(包括一条常常的垂直线和两条水平线)
# 水平线的长度是由聚类中的误差情况决定的。线条越长就越表明，合并在一起的两个聚类差别很大，而线条越短则越表明，两个聚类的相似度很高
def drawnode(draw, clust, x,y, scalling, labels):
    if clust.id<0:
        h1 = getheight(clust.left)*20
        h2 = getheight(clust.right)*20
        top = y-(h1+h2)/2
        bottom = y+(h1+h2)/2
        # 线的长度
        ll = clust.distance*scalling
        # 聚类到其子节点的垂直线
        draw.line((x,top+h1/2, x, bottom-h2/2), fill=(255,0,0))

        # 连接左侧节点的水平线
        draw.line((x,top+h1/2, x+ll, top+h1/2), fill=(255,0,0))

        # 连接右侧节点的水平线
        draw.line((x,bottom-h2/2, x+ll, bottom-h2/2), fill=(255,0,0))

        # 调用函数绘制左右结点
        drawnode(draw, clust.left, x+ll, top+h1/2, scalling, labels)
        drawnode(draw, clust.right, x+ll, bottom-h2/2, scalling, labels)
    else:
        # 如果这是一个叶节点，则绘制节点的标签
        draw.text((x+5, y-7), labels[clust.id], (0,0,0))

# drawdendrogram(clust, blognames, jpeg='blogclust.jpg')

def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

rdata = rotatematrix(data)
wordclust = hcluster(rdata)
drawdendrogram(wordclust, lables=words, jpeg='wordclust.jpg')