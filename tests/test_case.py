from supar import Parser
parser = Parser.load('biaffine-dep-en')
dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
print(dataset)
