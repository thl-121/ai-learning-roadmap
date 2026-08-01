#1.变量
name = "Tom"
age = 20
score = 88.5
is_student = True

print(name,age,score,is_student)


#2.列表
scores = [80,90,75,88,95]

print("最高分：",max(scores))
print("最低分: ",min(scores))
print("平均分: ",sum(scores)/len(scores))

#3.条件判断
if score >=90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")

#4.循环
for s in scores:
    if s  >=85:
        print(s,"高分")
    else:
        print(s,"普通")


#5.字典
student={
    "name":"Tom",
    "age":20,
    "major":"智能科学与技术",
    "score":88.5
}

print(student["name"])
print(student["score"])


#6.函数
def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >=70:
        return "C"
    elif score >=60:
        return "D"
    else:
        return "F"


for s in scores:
    print(get_grade(s))


#练习
student_score=[66,67,80,89,90]
print(max(student_score))
print(min(student_score))
print(sum(student_score)/len(student_score))
for s in student_score:
    if s >= 90:
        print("优秀")
    elif s >= 70:
        print("良好")
    elif s>= 60:
        print("及格")
    else:
        print("不及格")

def get_age(age):
    if age>=18:
        print("已成年")
    else:
        print("未成年")


def get_num(a,b):
    return a+b,a-b,a*b,a/b

print(get_num(10,9))
print(get_age(18))

student={
    "name":"thl",
    "age":20,
    "major":"智能科学与技术",
    "score":568
}
student1={
    "name":"thd",
    "age":20,
    "major":"工程",
    "score":555
}
all_student=[student,student1]
for m in all_student:
    print(m)

max_stu = all_student[0]
for m in all_student:
    if m["score"] > max_stu["score"]:
        max_stu=m
print("分数最高的学生:",max_stu)

