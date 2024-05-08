cd data_processing

git clone https://github.com/craigsapp/beethoven-piano-sonatas.git
git clone https://github.com/craigsapp/haydn-piano-sonatas.git
git clone https://github.com/craigsapp/mozart-piano-sonatas.git
git clone https://github.com/craigsapp/scarlatti-keyboard-sonatas.git
git clone https://github.com/pl-wnifc/humdrum-chopin-first-editions.git
git clone https://github.com/craigsapp/joplin.git

if [ ! -d "kern" ]; then
    mkdir kern
fi

# Move Beehoven's sonatas to the kern directory and rename each file
for file in "beethoven-piano-sonatas/kern"/*.krn; do
    filename=$(basename -- "$file")
    newfilename="beethoven#$filename"
    cp "$file" "kern/$newfilename"
done

# Move Haydn's sonatas to the kern directory and rename each file
for file in "haydn-piano-sonatas/kern"/*.krn; do
    filename=$(basename -- "$file")
    newfilename="haydn#$filename"
    cp "$file" "kern/$newfilename"
done

# Move Mozart's sonatas to the kern directory and rename each file
for file in "mozart-piano-sonatas/kern"/*.krn; do
    filename=$(basename -- "$file")
    newfilename="mozart#$filename"
    cp "$file" "kern/$newfilename"
done

# Move Scarlatti's sonatas to the kern directory and rename each file
for file in "scarlatti-keyboard-sonatas/kern"/*.krn; do
    filename=$(basename -- "$file")
    newfilename="scarlatti#$filename"
    cp "$file" "kern/$newfilename"
done

# Move Chopin's sonatas to the kern directory and rename each file
for file in "humdrum-chopin-first-editions/kern"/*.krn; do
    filename=$(basename -- "$file")
    newfilename="chopin#$filename"
    cp "$file" "kern/$newfilename"
done

# Move Joplin's sonatas to the kern directory and rename each file
for file in "joplin/kern"/*.krn; do
    filename=$(basename -- "$file")
    newfilename="joplin#$filename"
    cp "$file" "kern/$newfilename"
done