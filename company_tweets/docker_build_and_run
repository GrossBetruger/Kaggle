docker stop company_tweet
docker rm company_tweet
docker build -t  company_tweet .
docker volume create  company_tweet_db
docker run --name  company_tweet -p 5432:5432 -v company_tweet_db:/var/lib/postgresql/data -e POSTGRES_PASSWORD=password  company_tweet 
