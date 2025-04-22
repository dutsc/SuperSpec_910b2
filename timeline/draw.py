# 绘制活动时间图

import matplotlib.pyplot as plt
import numpy as np

label = []
map = {}
start = []
end = []
label_ = []
start_ = []
end_ = []

verify_start = []
verify_end = []
verify_time = []

log_file = "opt_tree_128_8"
prefixs = []

# with open('../logs/clean_time.log','r') as f:
with open('./logs/'+ log_file + ".log",'r') as f:
    lines = f.readlines()
    warm_up = False
    for i in range(len(lines)):
        if "warm up end" in lines[i]:
            warm_up = True
            continue
        if not warm_up:
            continue
        if "prefix_l" in lines[i]:
            prefix = float(lines[i].split(':')[-1])
            prefixs.append(prefix)
            
        if lines[i][0] == "-":
            label += label_
            start += start_
            end += end_
            label += ["*"]
            start += [111111111111111111111111111111111111.0]
            end += [111111111111111111111111111111111111.0]
            map = {}
            label_ = []
            start_ = []
            end_ = []
            continue
        if lines[i][0] == "#":
            line = lines[i][1:].split('@')
            label_.append(line[0].strip())
            key = line[0].strip()
            value = float(line[1].strip())
            if "llm forward" in key:
                verify_start.append(value)
            map[key] = len(label_) - 1
            start_.append(value)
            end_.append(0)
        elif lines[i][0] == "$":
            line = lines[i][1:].split('@')
            key = line[0].strip()
            value = float(line[1].strip())
            if "llm forward" in key:
                verify_end.append(value)
            end_[map[key]] = value
            
for i in range(len(verify_end)):
    verify_time.append(verify_end[i] - verify_start[i])
    
print(verify_time)
print(np.mean(verify_time))
print(f'{prefixs = }')
print(f'{np.mean(prefixs) = }')


# print(label)
# print(start)
# print(end)
# print(len(label))
# print(len(start))
# print(len(end))

start_time = min(start)
start = [i - start_time for i in start]
end = [i - start_time for i in end]

# print(len(start))
# print(len(end))

# 设置画布大小
# plt.figure(figsize=(15, 5))

fig = plt.figure(figsize=(50, 4))

# 将label和timestamps分成三组
label1s = []
label2s = []
label3s = []
label4s = []
start1s = []
start2s = []
start3s = []
start4s = []
end1s = []
end2s = []
end3s = []
end4s = []

label1 = []
label2 = []
label3 = []
label4 = []
start1 = []
start2 = []
start3 = []
start4 = []
end1 = []
end2 = []
end3 = []
end4 = []
for i in range(len(label)):
    if label[i] == "*":
        label1s.append(label1)
        label2s.append(label2)
        label3s.append(label3)
        label4s.append(label4)
        start1s.append(start1)
        start2s.append(start2)
        start3s.append(start3)
        start4s.append(start4)
        end1s.append(end1)
        end2s.append(end2)
        end3s.append(end3)
        end4s.append(end4)
        label1 = []
        label2 = []
        label3 = []
        label4 = []
        start1 = []
        start2 = []
        start3 = []
        start4 = []
        end1 = []
        end2 = []
        end3 = []
        end4 = []
        continue
    if "0" in label[i]:
        label1.append(label[i])
        start1.append(start[i])
        end1.append(end[i])
    elif "1" in label[i]:
        label2.append(label[i])
        start2.append(start[i])
        end2.append(end[i])
    elif "2" in label[i]:
        label3.append(label[i])
        start3.append(start[i])
        end3.append(end[i])
    else:
        label4.append(label[i])
        start4.append(start[i])
        end4.append(end[i])
    # else:
    #     label3.append(label[i])
    #     start3.append(start[i])
    #     end3.append(end[i])
        


# 条形图
def draw_barh(y, start, end):
    # colors = ['#63b2ee', '#76da91', '#f8cb7f', '#f89588', '#7cd6cf', '#9192ab', '#7898e1', '#efa666', '#eddd86', '#9987ce']
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'gray', 'black', 'brown', 'cyan', 'magenta']
    left = start
    width = [end[i] - start[i] for i in range(len(start))]
    plt.barh(y, width, left=left, height=0.5, color=colors)

for i in range(len(label1s)):
    draw_barh(label1s[i], start1s[i], end1s[i])
for i in range(len(label2s)):
    draw_barh(label2s[i], start2s[i], end2s[i])
for i in range(len(label3s)):
    draw_barh(label3s[i], start3s[i], end3s[i])
    # draw_barh(label4s[i], start4s[i], end4s[i])
    
total_time = 0
for i in range(len(label3s)):
    start = start3s[i]
    end = end3s[i]
    for j in range(len(start)):
        total_time += end[j] - start[j]
# print(total_time)
    

# draw_barh(label1, start1, end1)
# draw_barh(label2, start2, end2)
# draw_barh(label3, start3, end3)
# draw_barh(label4, start4, end4)


# 添加文本标签
# for i, txt in enumerate(epochs):
#     plt.annotate(f'{str(label[i])}', (epochs[i] + 0.1, timestamps[i]), textcoords="offset points", xytext=(5,0), ha='left')
#     plt.annotate(f'{str(round(data[i], 3))}', (epochs[i] - 0.1, timestamps[i]), textcoords="offset points", xytext=(5,0), ha='right')

# 设置x轴的刻度
# plt.xticks([0, 1, 2], [' ', ' ', ' '])
# plt.yticks([i for i in range(round(data[-1]) + 1)])

# plt.xlabel('Timestamp')
# plt.ylabel('Epoch')
# plt.title('Epochs Over Time')
# plt.xticks(rotation=45)
# plt.show()

fig.tight_layout()
 
plt.savefig('./images/'+ log_file +'.png')
print('done')