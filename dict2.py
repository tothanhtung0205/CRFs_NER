#-*- coding=utf-8 -*-
from io import open
import re

filter_non_str = re.compile(u'\w')

def read_file(file_name):
    prev_b_loc = []
    with open(file_name,'r',encoding='utf-8') as f:
        for line in f:
            x = line.split(' ')
            if filter_non_str.search(x[0]) != None:
                a = x[0]
                prev_b_loc.append(a)
    return set(prev_b_loc)


prev_b_loc = read_file('dataset/thong_ke/prev_b_loc.txt')
prev_b_org = read_file('dataset/thong_ke/prev_b_org.txt')
prev_b_per = read_file('dataset/thong_ke/prev_b_per.txt')

b_loc = read_file('dataset/thong_ke/b_loc.txt')
b_org = read_file('dataset/thong_ke/b_org.txt')
b_per = read_file('dataset/thong_ke/b_per.txt')

loc_giao_org = prev_b_loc.intersection(prev_b_org)
loc_giao_per = prev_b_loc.intersection(prev_b_per)
org_giao_per = prev_b_org.intersection(prev_b_per)

def maybe_is_loc(word):
    if word in b_loc:
        return True
    else:
        return False
def maybe_is_org(word):
    if word in b_org:
        return True
    else:
        return False
def maybe_is_per(word):
    if word in b_per:
        return True
    else:
        return False
def maybe_prev_loc(word):
    if word in prev_b_loc:
        return True
    else:
        return False
def maybe_prev_org(word):
    if word in prev_b_org:
        return True
    else:
        return False
def maybe_prev_per(word):
    if word in prev_b_per:
        return True
    else:
        return False

print maybe_is_org(u'công_ty')
