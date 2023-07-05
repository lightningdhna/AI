import os

label = os.listdir('data')


def get_att_by_num(num):
    s = label[int(num)]
    s.split(sep=' ')
    return [float(a) for a in s]


def get_finger_att(num, finger):
    s = label[int(num)]
    s.split(sep=' ')
    val = float(s[finger])
    return val


def get_name_by_att(att):
    s = []
    return ""


def get_label_from_att(att_hat):
    att = []
    return "{:1.1f}".format(att_hat)
    for i in att_hat:
        att.append("{:1.1f}".format(i))
    return str(att[0:5])
    # for i in att_hat:
    #     if i < 0.3:
    #         att.append(0)
    #     elif i > 0.7:
    #         att.append(1)
    #     else:
    #         return "Can't regconize"
    # if att[-1] == 1:
    #     return "Nothing"
    # return str(att[0:5])
