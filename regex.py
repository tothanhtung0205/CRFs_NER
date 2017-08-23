from io import open

ten_tinh = []
with open('dataset/tinh_thanh.txt','r',encoding='utf-8') as f:
    for line in f:
        line = line.replace(' ','_')
        line = line.encode('utf-8')
        ten_tinh.append(line)
    print ten_tinh

phan_cap_hanh_chinh = ["tỉnh" ,"thành_phố" ,"tp." ,"tp" ,"huyện" ,"quận","q.","xã" , "phường","p." , "thị_trấn" , "thôn" , "bản" , "làng" , "xóm","ấp","buôn"]
dia_chi = ["số","ngõ","ngách"]

chuc_vu = []