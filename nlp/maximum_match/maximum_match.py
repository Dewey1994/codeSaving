class Maximum_match:
    def __init__(self, dict_path):
        self.mp = set(i.strip() for i in open(dict_path, 'r', encoding='utf-8'))
        self.match_res = []
        self.max_length = len(max(self.mp, key=len))

    def maximum_match(self, sentence):
        sent_len = len(sentence)
        while sent_len > 0:
            divide = sentence[0:self.max_length]
            while divide not in self.mp:
                if len(divide) == 1:
                    break
                divide = divide[:len(divide) - 1]
            self.match_res.append(divide)
            sentence = sentence[len(divide):]
            sent_len = len(sentence)
        return self.match_res

    def reverse_maximum_match(self, sentence):
        sent_len = len(sentence)
        while sent_len > 0:
            divide = sentence[-self.max_length:]
            while divide not in self.mp:
                if len(divide) == 1:
                    break
                divide = divide[1:]
            self.match_res.append(divide)
            sentence = sentence[:-(len(divide))]
            sent_len = len(sentence)
        return self.match_res[::-1]



if __name__ == '__main__':
    ms = Maximum_match("dict.txt")
    res = ms.maximum_match("研究生命的起源")
    print(res)
