parent_a = 40
parent_b = 40

left_a = 30
left_b = 10

right_a = 10
right_b = 30

IgDparent = 1 - pow(parent_a/(parent_a+parent_b), 2) - \
    pow(parent_b/(parent_a+parent_b), 2)
print("Ig(Dparent) = " + str(IgDparent))
    
IgDleft = 1 - pow(left_a/(left_a+left_b), 2) - \
    pow(left_b/(left_a+left_b), 2)
print("Ig(Dleft) = " + str(IgDleft))

IgDright = 1 - pow(right_a/(right_a+right_b), 2) - \
    pow(right_b/(right_a+right_b), 2)
print("Ig(Dright) = " + str(IgDright))
    
information_gain = IgDparent - (left_a+left_b)/(parent_a+parent_b) * IgDleft \
    - (right_a+right_b)/(parent_a+parent_b) * IgDright
    
print("Information Gain = " + str(information_gain))