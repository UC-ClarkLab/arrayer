# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 08:32:56 2015

@author: Brian
"""

import string

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

with open('test_protocol_dirty.txt', 'r') as f:
    template = f.read()

# Position/Volume text file syntax:
# Treatment_#_[x/y/vol].txt
# e.g.
# Treatment_1_x.txt
# Treatment_7_y.txt
# Treatment_20_vol.txt

# Provide the output protocol file name
name_out = 'generated_protocol_v4'

# Provide the file path from the 'Brian/' base folder on the arrayer computer
path = 'TestPrint\Protocol'

# Gives the open time
t1_open = 3000
t2_open = 3000
t3_open = 3000
t4_open = 3000
t5_open = 3000
t6_open = 3000
t7_open = 3000
t8_open = 3000
t9_open = 3000
t10_open = 3000
t11_open = 3000
t12_open = 3000
t13_open = 3000
t14_open = 3000
t15_open = 3000
t16_open = 3000
t17_open = 3000
t18_open = 3000
t19_open = 3000
t20_open = 3000
t21_open = 3000
t22_open = 3000
t23_open = 3000
t24_open = 3000

# Loop values must equal the number of positions for each treatment condition
t1_loop = 84
t2_loop = 84
t3_loop = 84
t4_loop = 84
t5_loop = 84
t6_loop = 84
t7_loop = 84
t8_loop = 84
t9_loop = 84
t10_loop = 84
t11_loop = 84
t12_loop = 84
t13_loop = 84
t14_loop = 84
t15_loop = 84
t16_loop = 24
t17_loop = 36
t18_loop = 24
t19_loop = 42
t20_loop = 42
t21_loop = 42
t22_loop = 42
t23_loop = 42
t24_loop = 42

# Enabled must be 'TRUE' or 'FALSE'
t1_enabled = 'TRUE'
t2_enabled = 'TRUE'
t3_enabled = 'TRUE'
t4_enabled = 'TRUE'
t5_enabled = 'TRUE'
t6_enabled = 'TRUE'
t7_enabled = 'TRUE'
t8_enabled = 'TRUE'
t9_enabled = 'TRUE'
t10_enabled = 'TRUE'
t11_enabled = 'TRUE'
t12_enabled = 'TRUE'
t13_enabled = 'TRUE'
t14_enabled = 'TRUE'
t15_enabled = 'TRUE'
t16_enabled = 'TRUE'
t17_enabled = 'TRUE'
t18_enabled = 'TRUE'
t19_enabled = 'TRUE'
t20_enabled = 'TRUE'
t21_enabled = 'TRUE'
t22_enabled = 'TRUE'
t23_enabled = 'TRUE'
t24_enabled = 'TRUE'

l = [
    ('path', path),
    ('t1_open', t1_open),
    ('t2_open', t2_open),
    ('t3_open', t3_open),
    ('t4_open', t4_open),
    ('t5_open', t5_open),
    ('t6_open', t6_open),
    ('t7_open', t7_open),
    ('t8_open', t8_open),
    ('t9_open', t9_open),
    ('t10_open', t10_open),
    ('t11_open', t11_open),
    ('t12_open', t12_open),
    ('t13_open', t13_open),
    ('t14_open', t14_open),
    ('t15_open', t15_open),
    ('t16_open', t16_open),
    ('t17_open', t17_open),
    ('t18_open', t18_open),
    ('t19_open', t19_open),
    ('t20_open', t20_open),
    ('t21_open', t21_open),
    ('t22_open', t22_open),
    ('t23_open', t23_open),
    ('t24_open', t24_open),
    ('t1_loop', t1_loop),
    ('t2_loop', t2_loop),
    ('t3_loop', t3_loop),
    ('t4_loop', t4_loop),
    ('t5_loop', t5_loop),
    ('t6_loop', t6_loop),
    ('t7_loop', t7_loop),
    ('t8_loop', t8_loop),
    ('t9_loop', t9_loop),
    ('t10_loop', t10_loop),
    ('t11_loop', t11_loop),
    ('t12_loop', t12_loop),
    ('t13_loop', t13_loop),
    ('t14_loop', t14_loop),
    ('t15_loop', t15_loop),
    ('t16_loop', t16_loop),
    ('t17_loop', t17_loop),
    ('t18_loop', t18_loop),
    ('t19_loop', t19_loop),
    ('t20_loop', t20_loop),
    ('t21_loop', t21_loop),
    ('t22_loop', t22_loop),
    ('t23_loop', t23_loop),
    ('t24_loop', t24_loop),
    ('t1_enabled', t1_enabled),
    ('t2_enabled', t2_enabled),
    ('t3_enabled', t3_enabled),
    ('t4_enabled', t4_enabled),
    ('t5_enabled', t5_enabled),
    ('t6_enabled', t6_enabled),
    ('t7_enabled', t7_enabled),
    ('t8_enabled', t8_enabled),
    ('t9_enabled', t9_enabled),
    ('t10_enabled', t10_enabled),
    ('t11_enabled', t11_enabled),
    ('t12_enabled', t12_enabled),
    ('t13_enabled', t13_enabled),
    ('t14_enabled', t14_enabled),
    ('t15_enabled', t15_enabled),
    ('t16_enabled', t16_enabled),
    ('t17_enabled', t17_enabled),
    ('t18_enabled', t18_enabled),
    ('t19_enabled', t19_enabled),
    ('t20_enabled', t20_enabled),
    ('t21_enabled', t21_enabled),
    ('t22_enabled', t22_enabled),
    ('t23_enabled', t23_enabled),
    ('t24_enabled', t24_enabled)
    ]

d = dict([
    ('path', path),
    ('t1_open', t1_open),
    ('t2_open', t2_open),
    ('t3_open', t3_open),
    ('t4_open', t4_open),
    ('t5_open', t5_open),
    ('t6_open', t6_open),
    ('t7_open', t7_open),
    ('t8_open', t8_open),
    ('t9_open', t9_open),
    ('t10_open', t10_open),
    ('t11_open', t11_open),
    ('t12_open', t12_open),
    ('t13_open', t13_open),
    ('t14_open', t14_open),
    ('t15_open', t15_open),
    ('t16_open', t16_open),
    ('t17_open', t17_open),
    ('t18_open', t18_open),
    ('t19_open', t19_open),
    ('t20_open', t20_open),
    ('t21_open', t21_open),
    ('t22_open', t22_open),
    ('t23_open', t23_open),
    ('t24_open', t24_open),
    ('t1_loop', t1_loop),
    ('t2_loop', t2_loop),
    ('t3_loop', t3_loop),
    ('t4_loop', t4_loop),
    ('t5_loop', t5_loop),
    ('t6_loop', t6_loop),
    ('t7_loop', t7_loop),
    ('t8_loop', t8_loop),
    ('t9_loop', t9_loop),
    ('t10_loop', t10_loop),
    ('t11_loop', t11_loop),
    ('t12_loop', t12_loop),
    ('t13_loop', t13_loop),
    ('t14_loop', t14_loop),
    ('t15_loop', t15_loop),
    ('t16_loop', t16_loop),
    ('t17_loop', t17_loop),
    ('t18_loop', t18_loop),
    ('t19_loop', t19_loop),
    ('t20_loop', t20_loop),
    ('t21_loop', t21_loop),
    ('t22_loop', t22_loop),
    ('t23_loop', t23_loop),
    ('t24_loop', t24_loop),
    ('t1_enabled', t1_enabled),
    ('t2_enabled', t2_enabled),
    ('t3_enabled', t3_enabled),
    ('t4_enabled', t4_enabled),
    ('t5_enabled', t5_enabled),
    ('t6_enabled', t6_enabled),
    ('t7_enabled', t7_enabled),
    ('t8_enabled', t8_enabled),
    ('t9_enabled', t9_enabled),
    ('t10_enabled', t10_enabled),
    ('t11_enabled', t11_enabled),
    ('t12_enabled', t12_enabled),
    ('t13_enabled', t13_enabled),
    ('t14_enabled', t14_enabled),
    ('t15_enabled', t15_enabled),
    ('t16_enabled', t16_enabled),
    ('t17_enabled', t17_enabled),
    ('t18_enabled', t18_enabled),
    ('t19_enabled', t19_enabled),
    ('t20_enabled', t20_enabled),
    ('t21_enabled', t21_enabled),
    ('t22_enabled', t22_enabled),
    ('t23_enabled', t23_enabled),
    ('t24_enabled', t24_enabled)
    ])

protocol = string.Formatter().vformat(template, (), SafeDict(d))

'''
protocol = template.format(
    path=path,
    t1_open=t1_open,
    t2_open=t2_open,
    t3_open=t3_open,
    t4_open=t4_open,
    t5_open=t5_open,
    t6_open=t6_open,
    t7_open=t7_open,
    t8_open=t8_open,
    t9_open=t9_open,
    t10_open=t10_open,
    t11_open=t11_open,
    t12_open=t12_open,
    t13_open=t13_open,
    t14_open=t14_open,
    t15_open=t15_open,
    t16_open=t16_open,
    t17_open=t17_open,
    t18_open=t18_open,
    t19_open=t19_open,
    t20_open=t20_open,
    t21_open=t21_open,
    t22_open=t22_open,
    t23_open=t23_open,
    t24_open=t24_open,
    t1_loop=t1_loop,
    t2_loop=t2_loop,
    t3_loop=t3_loop,
    t4_loop=t4_loop,
    t5_loop=t5_loop,
    t6_loop=t6_loop,
    t7_loop=t7_loop,
    t8_loop=t8_loop,
    t9_loop=t9_loop,
    t10_loop=t10_loop,
    t11_loop=t11_loop,
    t12_loop=t12_loop,
    t13_loop=t13_loop,
    t14_loop=t14_loop,
    t15_loop=t15_loop,
    t16_loop=t16_loop,
    t17_loop=t17_loop,
    t18_loop=t18_loop,
    t19_loop=t19_loop,
    t20_loop=t20_loop,
    t21_loop=t21_loop,
    t22_loop=t22_loop,
    t23_loop=t23_loop,
    t24_loop=t24_loop,
    t1_enabled=t1_enabled,
    t2_enabled=t2_enabled,
    t3_enabled=t3_enabled,
    t4_enabled=t4_enabled,
    t5_enabled=t5_enabled,
    t6_enabled=t6_enabled,
    t7_enabled=t7_enabled,
    t8_enabled=t8_enabled,
    t9_enabled=t9_enabled,
    t10_enabled=t10_enabled,
    t11_enabled=t11_enabled,
    t12_enabled=t12_enabled,
    t13_enabled=t13_enabled,
    t14_enabled=t14_enabled,
    t15_enabled=t15_enabled,
    t16_enabled=t16_enabled,
    t17_enabled=t17_enabled,
    t18_enabled=t18_enabled,
    t19_enabled=t19_enabled,
    t20_enabled=t20_enabled,
    t21_enabled=t21_enabled,
    t22_enabled=t22_enabled,
    t23_enabled=t23_enabled,
    t24_enabled=t24_enabled)
'''
# Write protocol to file
with open('{0}.txt'.format(name_out), 'w') as h:
    h.write(protocol)

