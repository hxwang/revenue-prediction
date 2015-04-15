
score = load('.\data\score.txt')
[rank, idx] = sort(score, 'ascend')

 for i =1:1:size(idx,1)
  histScore(num2str(idx(i)),  score(idx(i),1), num2str(i))
 end