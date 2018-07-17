#!/bin/bash
# filename='bigfile.txt'
# r = requests.put('%s/%s' % (bucket_url,filename),
#                  data=open(filename, 'rb'),
#                  headers={"Accept":"application/json",
#                           "Authorization":"Bearer %s" % ACCESS_TOKEN,
#                           "Content-Type":"application/octet-stream"})

id=$1
file=$2

url=https://zenodo.org/api/deposit/depositions/
bucket=$(curl "$url/$1?access_token=$ZENODO_TOKEN" | jq '.links.bucket')
name=$(basename file)
bucket=$bucket


curl  -H "Authorization: Bearer $ZENODO_TOKEN" -H "Accept: application/json" -H "Content-Type: application/octet-stream" \
      --progress-bar \
      -o ${name}.curl.out \
      -X PUT $bucket/$basename -T $file
