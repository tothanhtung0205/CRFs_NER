#-*- coding=utf-8 -*-

from io import open

ho = []
with open ('dataset/ho_nguoi_viet.txt','r',encoding='utf-8') as f:
    for line in f:
        line = line.encode('utf-8')
        line = line.rstrip()
        ho.append(line)

phan_cap_hanh_chinh = ["số","ngõ","ngách","tỉnh" ,"thành_phố" ,"tp." ,"tp" ,"huyện" ,"quận","q.","xã" , "phường","p." , "thị_trấn" , "thôn" , "bản" , "làng" , "xóm","ấp","buôn"]
bao_chi = ["báo","tờ","tạp_chí","đài","thông_tấn_xã"]
cap_bac_hoc = ["mầm_non","tiểu_học","th","trung_học","thcs","thpt","trung_cấp","cao_đẳng","cđ","đại_học","đh"]
nghien_cuu = ["trường","viện","học_viện"]
cong_ty = ["công_ty","tập_đoàn","hãng","xí_nghiệp"]
bo_may_nha_nuoc = ["sở","phòng","ban","cục","chi_cục","tổng_cục","vụ","bộ"]
chuc_vu = ["tổng_thống","chủ_tịch","bí_thư","thủ_tướng","bộ_trưởng","thứ_trưởng","viện_trưởng","chánh_án","thống_đốc","giám_đốc","trưởng_phòng"]
dai_tu_nhan_xung = ['anh','chị','em','ông','bà','cô','gì','chú','bác','cậu','mợ','ngài','vị']
tinh_thanh = ["an_giang","bà_rịa_vũng_tàu","vũng_tàu","bắc_giang","bắc_kạn","bạc_liêu","bắc_ninh","bến_tre","bình_định","bình_dương","bình_phước","bình_thuận","cà_mau","cao_bằng","đắk_lắk","đắk_nông","điện_biên","đồng_nai","đồng_tháp","gia_lai","hà_giang","hà_nam","hà_tĩnh","hải_dương","hậu_giang","hòa_bình","hưng_yên","khánh_hòa","kiên_giang","kon_tum","lai_châu","lâm_đồng","lạng_sơn","lào_cai","long_an","nam_định","nghệ_an","ninh_bình","ninh_thuận","phú_thọ","quảng_bình","quảng_nam","quảng_ngãi","quảng_ninh","quảng_trị","sóc_trăng","sơn_la","tây_ninh","thái_bình","thái_nguyên","thanh_hóa","thừa_thiên_huế","tiền_giang","trà_vinh","tuyên_quang","vĩnh_long","vĩnh_phúc","yên_bái","phú_yên","cần_thơ","đà_nẵng","hải_phòng","hà_nội","tp_hcm"]

def is_full_name(word):
    temp = word.split('_')
    if temp[0] in ho:
        return True
    else:
        return False
def get_regex(word):

    if word in phan_cap_hanh_chinh:
        regex = "phan_cap_hanh_chinh"
    elif word in bao_chi:
        regex = "bao_chi"
    elif word in cap_bac_hoc:
        regex = "cap_bac_hoc"
    elif word in nghien_cuu:
        regex = "nghien_cuu"
    elif word in cong_ty:
        regex = "cong_ty"
    elif word in bo_may_nha_nuoc:
        regex = "bo_may_nha_nuoc"
    elif word in chuc_vu:
        regex = "chuc_vu"
    elif word in tinh_thanh:
        regex = "tinh_thanh"
    elif is_full_name(word):
        regex = "ten nguoi"
    elif word in dai_tu_nhan_xung:
        regex = "dtnx"
    else:
        regex = "na"
    return regex
