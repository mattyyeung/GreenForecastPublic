# Greenforecast.au
7 day outlook of Renewable Electricity and Power Price in the National Electricity Market, powered by AI

## Repository Structure

### Components
The 4 components are largely standalone; ie they are not intended to be subpackages. Each has run on different hardware. 
- Dataset generator: creates the training sets. Downloads info from Aemo, oikolab, etc.
- training: Trains all the models. Runs on something with a decent graphics card.
- forecaster: Runs periodically to generate the forecasts. Runs on AWS lambda.
- website: static html/css/js that displays data, housed on S3.
![Deployment diagram](/deployment_diagram.png)

### Dependencies
- dataset generator: 
    - juptyer lab
- training:
    - a GPU
    - other depencies pulled in when the virtual environment is created, see below
- forecaster:
    - Docker - for deploying the forecaster to AWS
        - Docker for windows is free for personal use
        - used WSL2 on Windows. 
            - [Instructions to set up docker without docker desktop (WSL2)](https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9)
            - Account user/pass is mxy/mxy. 
            - need to run WSL as admin!
            - had to run `wsl --set-version Ubuntu 2` in powershell because it defaulted to wsl2 ?? check with `wsl --list --verbose`
            - project folder is in /mnt/c/path/to/green_forecast/
            - had a certificate issue with zscaler. got a x.509 certificate error when docker tried to download the aws image. Turns out this was a wsl2 certificate issue; it didn't have the zscaler cert on it, which was MITM'ing all SSL as it does. reproduced an error with `wget https://public.ecr.aws/v2/`. So saved zscaler cert using chrome to a base64, renamed as .pem, then moved that file to /usr/local/share/ca-certificates/ on ubuntu wsl. Then ran `sudo update-ca-certificates --fresh`. (This)[https://stackoverflow.com/questions/31205438/docker-on-windows-boot2docker-certificate-signed-by-unknown-authority-error] and [this](https://community.zscaler.com/t/installing-tls-ssl-root-certificates-to-non-standard-environments/7261) helped. 
    - make - On windows: `choco install make`... ut that din't woek because writing x-platform makefiles sucks.
    - conda (which comes with miniconda or anaconda) for making virtual environments
    - other dependencies are in `requirements.txt` but are automatically fetched, see below

### Virtual environment for training or general mucking around:
Create virtual environment: (run this from the same directory as this file, which has a requirements.txt)
```conda create --name mxy-py39 --file requirements.txt python=3.9```

activate that environment:
```conda activate mxy-py39```

this should run jupyter from WITHIN that environment... 
```jupyter lab``` ... if that doesn't play nice, could run this to make sure it uses the right version of python/jupyter: ```python -m jupyter lab```

When done: ```conda deactivate```

### Virtual environment for forecaster
This uses `forecaster/requirements.txt` not the one in the root directory. This is specifically for forecasting to emulates the environment used on lambda

Easiest to run this from docker, see below. But also did the venv locally:
- `cd forecaster` from project root directory
- `conda create --name mxy-forecaster --file requirements_forecaster.txt python=3.9`
- `conda activate mxy-forecaster`
I don't think that will have jupyter installed, so could run `conda install jupyter` and then `python -m jupyter lab` from within that? (untested)

Once that's set up, run `make` to see the possible options for running / setting up the forecaster. 

### Container image for forecaster:
Container image so can test forecaster locally, specifically to emulate lambda where there is a read-only filesystem and no gpu available. 

Run `make` from root directory to see the various options for forecaster, these all start with `make fc_XXXX`. 

### Website - Setting up AWS 

Followed [this guide](https://support.cloudflare.com/hc/en-us/articles/360037983412-Configuring-an-Amazon-Web-Services-static-site-to-use-Cloudflare) mostly... but the bucket policy was wrong.

AWS Buckets:
- greenforecast.au  
	- http://greenforecast.au.s3-website-ap-southeast-2.amazonaws.com/
	- Bucket policy is set for greenforecast.au so only cloudflare IP addresses are allowed to access it. 
- www.greenforecast.au:
	- redirects to greenforecast.au
	- http://www.greenforecast.au.s3-website-ap-southeast-2.amazonaws.com/ 

Domain: greenforecast.au with ventraip.com.au. Nameservers: MAGALI.NS.CLOUDFLARE.COM and VIDDY.NS.CLOUDFLARE.COM

Cloudflare: 
- handles SSL - S3 doesn't support HTTPS
- DNS host
- caching is a bonus
- could have used cloudfront but just felt like using cloudflare.

Deploying: see makefile. 

### Forecaster - AWS

AWS ECR:
From [instructions](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html):
- Ran this to give docker aws credentials:
```aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 856549827017.dkr.ecr.ap-southeast-2.amazonaws.com```
- Make ECR repository:
```aws ecr create-repository --repository-name greenforecast_ecr --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE```
- Tag the docker image and then upload it: use `make fc_deploy`. This doesn't delete the old image (but it does remove the 'latest' tag, leaving the old image untagged). Set up a [lifecycle rule](https://docs.aws.amazon.com/AmazonECR/latest/userguide/LifecyclePolicies.html) to delte old images.

To get the container working:
- lambda is read-only filesystem, added --readonly and --tmpfs /tmp to docker run to emulate this; changed written folders to /tmp
- added IAM role greenforecast_forecaster-role-jhv5py used this for the lambda function execution role, gave permissions to read/write to S3 buckets.
- need to get credentials into the image when running locally - pass thru env vars with docker run. Handled automatically when on lambda.
- had 'no space left on device' errors.. docker system prune and giving more disk space in docker GUI. 

EventBridge:
lambda involked every 2hours, at 5mins past the hour (guessing 5 mins is good to give NEM data time to update). used [this guide](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-run-lambda-schedule.html)
