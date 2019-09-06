function Mat = myKr2(A,B,partA,partB)

partionNum=size(partB,2);
temp=(zeros(size(A,1)*size(B,1),partionNum));
for i=1:partionNum
    t=A(:,sumRank(partA,i-1)+1:sumRank(partA,i))*B(:,sumRank(partB,i-1)+1:sumRank(partB,i))';
    temp(:,i)= t(:);
end
Mat=temp;
end