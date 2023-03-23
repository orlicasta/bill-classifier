# fetch only from $branch and then set current branch to $branch
git checkout $1
git fetch origin $1
        
# if there are any changes in the current branch other than to the README.md, then pull changes to current branch, train model, and update model params
if [[ $(git diff $1 origin/$1 --name-only | grep -v -e README.md | wc -l) -gt 0 ]] 
then
git merge FETCH_HEAD
cd ..
py src/cnn.py
git add params.pt
git commit -m "params change after training"
git push origin $1
fi
