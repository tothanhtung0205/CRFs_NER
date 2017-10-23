from io import open
import re


filter_non_str = re.compile(u'\w')

def read_file(file_name):
    prev_b_loc = []
    with open(file_name,'r',encoding='utf-8') as f:
        for line in f:
            x = line.split(' ')
            if filter_non_str.search(x[0]) != None:
                prev_b_loc.append(x[0])
    return set(prev_b_loc)

prev_b_loc = read_file('thong_ke/prev_b_loc.txt')
prev_b_org = read_file('thong_ke/prev_b_org.txt')
prev_b_per = read_file('thong_ke/prev_b_per.txt')

giao = prev_b_loc.intersection(prev_b_org)
giao2 = prev_b_loc.intersection(prev_b_per)
giao3 = prev_b_org.intersection(prev_b_per)
print prev_b_loc

