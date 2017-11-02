import  feedparser
import  re

# 返回一个RSS订阅源的标题和包含单词计数情况的字典
def getwordcounts(url):
    # 解析订阅源
    d = feedparser.parse(url)
    wc = {}

    # 循环遍历所有的文章条目
    for e in d.entries:
        if 'summary' in e: summary = e.summary
        else: summary = e.description

        # 提取一个单词列表
        words = getwords(e.title + ' ' + summary)
        for word in words:
            wc.setdefault(word, 0)
            wc[word] += 1
    return d.feed.items,wc


def getwords(html):
    # Remove all the HTML tags
    txt = re.compile(r'<[^>]+>').sub('', html)
    # Split words by all non-alpha characters
    words = re.compile(r'[^A-Z^a-z]+').split(txt)
    # Convert to lowercase
    return [word.lower() for word in words if word != '']


# 循环遍历订阅源并生成数据集
apcount = {}
wordcounts = {}
feedlist = [line for line in open('./data/feedlist.txt','r')]
for feedurl in feedlist:
    title,wc=getwordcounts(feedurl)
    wordcounts[title] = wc
    for word, count in wc.items():
        apcount.setdefault(word, 0)
        if count > 1:
            apcount[word] += 1

# 建立单词列表，用于针对每个博客的单词计数。并分别取10%和50%作为上下界
wordlist = []
for w,bc in apcount.items():
    frac = float(bc)/len(feedlist)
    if frac > 0.1 and frac < 0.5: wordlist.append(w)
# 记录针对每个博客的所有单词的统计情况
out = open('myblogdata.txt', 'w')
out.write('Blog')
for word in wordlist: out.write('\t%s' % word)
out.write('\n')
for blog, wc in wordcounts.items():
    out.write(blog)
    for word in wordlist:
        if word in wc: out.write('\t%d' % wc[word])
        else: out.write('\t0')
    out.write('\n')