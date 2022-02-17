#!/bin/bash
# create a folder to store the downloaded files
if [! -d MIME]; then mkdir MIME; fi
# go to the folder
cd MIME
# define and array with the download links
declare -a links=(
"https://www.dropbox.com/sh/wmyek0jhrpm0hmh/AADypxdDq9CbeD4VRKulTbG1a?dl=0>1.zip"
"https://www.dropbox.com/sh/kfxdbxy1dsju79i/AAD_0YyCm17oradIgtoTEidja?dl=0>2.zip"
"https://www.dropbox.com/sh/q33ko971usb028x/AACZJ25JmluVCeDU7ZrnMRqNa?dl=0>3.zip"
"https://www.dropbox.com/sh/dofxnu3ncum39hs/AAD2GKHwT7OtA4rtYxCnViNKa?dl=0>4.zip"
"https://www.dropbox.com/sh/z3o7ad8jt7ji8h1/AAC4qTLuAyYq5EXccBUuuzt6a?dl=0>5.zip"
"https://www.dropbox.com/sh/grgj9f8zyw4irfh/AAA2BKWxRpXgyy8R96bRRU1ua?dl=0>6.zip"
"https://www.dropbox.com/sh/hb2rxqnjpcx2591/AADU0alwAPCxRY0qfNS_O0Cza?dl=0>7.zip"
"https://www.dropbox.com/sh/uj7v0axgceddgw7/AAC4GB-uPSgtrPsJpAfNYjJta?dl=0>8.zip"
"https://www.dropbox.com/sh/1sj84icjhrvxwf4/AADOg_NUFI_oU6lR3E0mZhkQa?dl=0>9.zip"
"https://www.dropbox.com/sh/99jyhxvvs1cwqt4/AAB4zokOUUA2x2QIEGrZUawNa?dl=0>10.zip"
"https://www.dropbox.com/sh/m9emo93c0nwl23k/AACb912BIksbG1PWHO0T8GMYa?dl=0>11.zip"
"https://www.dropbox.com/sh/61s1x4f6yrjj2fp/AABBHku0_4EGyXH3__vAyijTa?dl=0>12.zip"
"https://www.dropbox.com/sh/hrwbreflwsy3x2b/AACDSbpiCRUIwfNWjI4lyydya?dl=0>13.zip"
"https://www.dropbox.com/sh/i1yacr9dzuxzmy3/AACG-uJ6OYLwvwC0RngIMI5ja?dl=0>14.zip"
"https://www.dropbox.com/sh/wqqrf0fjgpxseld/AADln-EXkyVCCRU_XM75qlfSa?dl=0>15.zip"
"https://www.dropbox.com/sh/jsrxsxm9em42uj2/AACgj6_NIjxJC1dKgK35TsJSa?dl=0>16.zip"
"https://www.dropbox.com/sh/qph8y8kwsu470vx/AACBW2NQ1WAcj227GWm1wYfoa?dl=0>17.zip"
"https://www.dropbox.com/sh/ui2rcfdu3jbb4a4/AAAlwwPFXWd-MggI62uEwCwna?dl=0>18.zip"
"https://www.dropbox.com/sh/6ysn4cbq5uuxcij/AAAAjU_yA79odQMGgYtVWJA6a?dl=0>19.zip"
"https://www.dropbox.com/sh/x60fptqni9cvqx8/AAB71YePoQfovylmMzZILWSxa?dl=0>20.zip"
)
# download the files requested by the user in arguments
for i in "$@"
do
    # download the file if it is not already in the folder and there is no folder with the same name
    if [ ! -f "$i.zip" ] && [ ! -d "$i" ]; then
        wget "${links[$i]}"
        # extract the file
        unzip "$i.zip"
        # remove the zip file
        rm "$i.zip"
    fi
done
# go back to the previous folder
cd ..