query = [11, 924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]

dic = {}
with open('q1_out.txt', 'r') as f:
    for line in f:
        line = line.strip()
        user, friends = line.split('\t')
        user = int(user)
        friends = [int(f) for f in friends.split(',')]
        dic[user] = friends

for user in query:
    print(user, '\t', dic[user])