pcc<-function(element,boundaries,dat,reference,predicted,wepal,wepalinp,sav,loc,nam=element){
  #element = total_nitrogen
  #boundaries = numeric vector [10, 15, 20]
  #dat = data.frame with input data predictions and wetchem and sample codes
  #reference = column number of object (numeric eg. 12)
  #predicted = column number of object where predictions are in
  #wepal = FALSE
  #wepalinp = 
  #sav = FALSE
  ## Check what was given in function body:
  
  # Check on element-object:
  
  if(exists("element")==F){
  warning("Without defining an element name in element-object the output will be without element name!")
  }
  if(exists("element")==T){
    if(class(element)!="character"){
      warning("Class of element has to be character! It will be coerced to class character.")
      element<-as.character(element)
    }
  }
  
  # Check on boundaries-object:
  
  if(exists("boundaries")==F){
    stop("Without defining the element boundaries in boundaries-object no output can be generated!")
  }
  if(exists("boundaries")==T){
    if(class(boundaries)!="numeric"){
      warning("Class of element-object has to be numeric! It will be coerced to class numeric.")
      boundaries<-as.numeric(boundaries)
      if(length(is.na(boundaries)==T)==length(boundaries)){
        stop("boundaries-object can't be coerced to class numeric!")
      }
    }
  }
  
  # Check on dat-object:
  
  if(exists("dat")==F){
    stop("Without defining the dat-object no output can be generated!")
  }
  if(exists("dat")==T){
    if(class(dat)!="data.frame" & class(dat)!="matrix"){
      stop("Class of dat-object has to be data.frame or matrix!")
    }
    if(class(dat)=="matrix"){
      dat<-as.data.frame(dat)
    }
  }
  
  # Check on reference-object:
  
  if(exists("reference")==F){
    stop("Without defining the reference-object no output can be generated!")
  }
  if(exists("reference")==T){
    if(length(reference)>1){
      stop("reference-object has to be of length 1!")
    }
    if(class(reference)!="numeric"){
     warning("Class of reference-object has to be numeric! It will be coerced to class numeric.")
      reference<-as.numeric(reference)
      if(length(reference)==1 & is.na(reference)==T){
        stop("reference-object can't be coerced to class numeric!")
      }
    }
    if(is.na(colnames(dat)[reference])==T){
      stop("reference-object does not contain meaningful information.")
    }
  }
  
  # Check on predicted-object:
  
  if(exists("predicted")==F){
    stop("Without defining the predicted-object no output can be generated!")
  }
  if(exists("predicted")==T){
    if(length(predicted)>1){
      stop("predicted-object has to be of length 1!")
    }
    if(class(predicted)!="numeric"){
      warning("Class of predicted-object has to be numeric! It will be coerced to class numeric.")
      predicted<-as.numeric(predicted)
      if(length(predicted)==1 & is.na(predicted)==T){
        stop("predicted-object can't be coerced to class numeric!")
      }
    }
    if(is.na(colnames(dat)[predicted])==T){
      stop("predicted-object does not contain meaningful information.")
    }
  }
  
  # Check on wepal- and wepalinp-object:
  
  if(exists("wepal")==F){
    warning("If wepal-object is not defined the relation of the element RMSECV to WEPAL standard deviation will not be calculated.")
  }
  if(exists("wepal")==T){
    if(wepal!=TRUE & wepal!=FALSE){
      warning("wepal-object has to be either 'TRUE' or 'FALSE'. The relation of the element RMSECV to WEPAL standard deviation will not be calculated.")
      if(wepal==TRUE){
        
        # Check on wepalinp-object:
        
        if(exists("wepalinp")==F){
          stop("Without defining the wepalinp-object no output for WEPAL-relation can be generated!")
        }
        if(exists("wepalinp")==T){
          if(class(wepalinp)!="data.frame" & class(wepalinp)!="matrix"){
            stop("Class of wepalinp-object has to be data.frame or matrix!")
          }
          if(class(wepalinp)=="matrix"){
            wepalinp<-as.data.frame(wepalinp)
          }
          if(length(na.omit(match(c("wepalMean","wepalSd"),colnames(wepalinp))))!=2){
            stop("Colnames of wepalinp-object have to equal 'wepalMean' and 'wepalSd'.")
          }
        }
      }
    }  
  }
    
  # Check on sav-object:
  
  if(exists("sav")==F){
    warning("If sav-object is not defined no plot will be stored on local drive. A plot is generated only.")
  }
  if(exists("sav")==T){
    if(sav!=TRUE & sav!=FALSE){
      warning("sav-object has to be either 'TRUE' or 'FALSE'. No plot will be stored on local drive. A plot is generated only.")
    }  
  }
  
  # Check on nam-object:
  if((class(nam)!="character" | length(nam)!=1) & sav==T){
    if (file.access(loc)==0) {
      warning("nam-object is either not of class 'character' or not of length one. 'element'-object will be used as file name.")
    }
  }
  
  ## Create classes for percent correct classified:
  
  b<-matrix(nrow=length(boundaries)+1,ncol=4,dimnames=list(paste("class ",c(1:(length(boundaries)+1)),sep=""),c("low","high","correctlyClassified","n")))
  b<-as.data.frame(b)
  for(i in 1:nrow(b)){
    if(i==1){
      b[i,1]<-(-Inf)
      b[i,2]<-boundaries[1]
    }
    if(i>1 & i<nrow(b)){
      b[i,1]<-boundaries[i-1]
      b[i,2]<-boundaries[i]
    }
    if(i==nrow(b)){
      b[i,1]<-boundaries[i-1]
      b[i,2]<-Inf
    }
  }
  
  ## Calculate per class percent correctly classified (short version):
  
  sum<-c()
  n=c()
  mi<-round(min(dat[,reference]),0)
  ma<-round(max(dat[,reference]),0)
  for(i in 1:nrow(b)){
    bla<-which(dat[,reference]>=b[i,1] & dat[,reference]<b[i,2])
    n[i]=length(bla)
    if(length(bla)>0){
      blo<-which(dat[bla,predicted]>=b[i,1] & dat[bla,predicted]<b[i,2])
      if(length(blo)>0){
        sum[i]<-round(length(blo)/length(bla)*100,0)  
      }
      if(length(blo)==0){
        sum[i]<-0
      }
    }
    if(length(bla)==0){
      sum[i]<-"no sample in this class"
    }  
  }
  b[,3]<-sum
  b[,4]=n
  
  ## Calculate full matrix of expected vs observed classified into (extended version):
    
  # Prepare output object (confusion matrix):
  
  bla<-nrow(b)
  ext<-as.data.frame(matrix(nrow=bla,ncol=bla,dimnames=list(paste("expected ",rownames(b),sep=""),paste("predicted ",rownames(b),sep=""))))
  
  # Fill output object:
  
  for(i in 1:nrow(b)){
    bla<-which(dat[,reference]>=b[i,1] & dat[,reference]<b[i,2])
    if(length(bla)>0){
      for(k in 1:nrow(b)){
        blo<-which(dat[bla,predicted]>=b[k,1] & dat[bla,predicted]<b[k,2])
        if(length(blo)>0){
          ext[i,k]<-signif(length(blo)/length(bla)*100,2)  
        }
        if(length(blo)==0){
          ext[i,k]<-0
        }
      } 
    }
    if(length(bla)==0){
      ext[i,1:ncol(ext)]<-"no sample in classExpected"
    }
  }  
  
  # Calculate accuracy, pcc1...:
  
  pecocl=as.data.frame(matrix(nrow=1,ncol=4,dimnames=list(element,c("Accuracy","PCC1","PCC2","PCC3"))))
  su=sum(b$n)
  a1=as.numeric(b$correctlyClassified[1])/100*b$n[1]/su*100
  a2=as.numeric(b$correctlyClassified[2])/100*b$n[2]/su*100
  a3=as.numeric(b$correctlyClassified[3])/100*b$n[3]/su*100
  a4=as.numeric(b$correctlyClassified[4])/100*b$n[4]/su*100
  pecocl$Accuracy=signif(sum(a1,a2,a3,a4,na.rm=T),2)
  #pecocl$Accuracy=signif(a1+a2+a3+a4,2)
  a1=as.numeric(ext[1,2])/100*b$n[1]/su*100
  a2=(as.numeric(ext[2,1])+as.numeric(ext[2,3]))/100*b$n[2]/su*100
  a3=(as.numeric(ext[3,2])+as.numeric(ext[3,4]))/100*b$n[3]/su*100
  a4=as.numeric(ext[4,3])/100*b$n[4]/su*100
  pecocl$PCC1=signif(sum(a1,a2,a3,a4,na.rm=T),2)
  #pecocl$PCC1=signif(a1+a2+a3+a4,2)
  a1=as.numeric(ext[1,3])/100*b$n[1]/su*100
  a2=as.numeric(ext[2,4])/100*b$n[2]/su*100
  a3=as.numeric(ext[3,1])/100*b$n[3]/su*100
  a4=as.numeric(ext[4,2])/100*b$n[4]/su*100
  pecocl$PCC2=signif(sum(a1,a2,a3,a4,na.rm=T),2)
  #pecocl$PCC2=signif(a1+a2+a3+a4,2)
  
  a1=as.numeric(ext[1,4])/100*b$n[1]/su*100
  a4=as.numeric(ext[4,1])/100*b$n[4]/su*100
  pecocl$PCC3=signif(sum(a1,a4,na.rm=T),2)
  #pecocl$PCC3=signif(a1+a4,2)
  
  ## Calculate RMESCV in 4 quartile classes:
  
  # Create output object:
  
  rms<-matrix(nrow=4,ncol=3,dimnames=list(paste("quartile ",c(1:4),sep=""),c("low","high","classRMSECV")))
  rms<-as.data.frame(rms)
  bou<-as.numeric(summary(dat[,reference])[c(1,2,3,5,6)])
  for(i in 1:nrow(rms)){
    rms[i,1:2]<-bou[i:(i+1)]
  }  
     
  # Fill object:
  
  for(i in 1:nrow(rms)){
    bla<-which(dat[,reference]>=rms[i,"low"] & dat[,reference]<=rms[i,"high"])
    rms[i,"classRMSECV"]<-signif(sqrt(sum((dat[bla,reference]-dat[bla,predicted])^2)/length(bla)),3)
  }
  
  ## Calculate the WEPAL relation:
  
  if(wepal==F){
    wep<-NA
  }
  if(wepal==T){
    print("Blah")
    wep<-rms
    colnames(wep)[ncol(wep)]<-c("WEPALRelation")
    wep$WEPALRelation<-NA
    for(i in 1:nrow(wep)){
      bla<-which(we[,"wepalMean"]>=b[i,1] & we[,"wepalMean"]<b[i,2])
      if(length(bla)>0){
       bb<-sqrt(sum(we[bla,"wepalSd"]^2)/length(bla))
      wep[i,ncol(wep)]<-round(rms[i,"classRMSECV"]/bb,1)
      }
      if(length(bla)==0){
        wep[i,ncol(wep)]<-"no WEPAL sample in this range"
      }
    } 
  }
  
  ## Adapt b-, rms- and wep-object:
  
  b[1,1]<-mi
  b[nrow(b),2]<-ma
  rms[1,1]<-mi
  rms[nrow(rms),2]<-ma
  #rms[,1:2]<-round(rms[,1:2],0)
  if (!is.na(wep)) {
    wep[1,1]<-mi
    wep[nrow(wep),2]<-ma
    wep[,1:2]<-round(wep[,1:2],0)
  }
  
  ## Calculate corrected accuracy:
  
  ca=ext
  
  # Penalize wrong predictions (pcc1=*2, pcc2=*3, pcc3=*4; overprediction double value)
  
  for(i in 1:nrow(ca)){
    if(nrow(ca)==4){
      if(i==1){
        fa=c(1,3,6,12)#fa=c(1,4,6,8,10)
      }
      if(i==2){
        fa=c(1.5,1,3,6)#fa=c(2,1,4,6,8)
      }
      if(i==3){
        fa=c(3,1.5,1,3)#fa=c(3,2,1,4,6)
      }
      if(i==4){
        fa=c(6,3,1.5,1)#fa=c(4,3,2,1,4)
      }
    }
    if(nrow(ca)==5){
      if(i==1){
        fa=c(1,3,6,12,24)#fa=c(1,4,6,8,10)
      }
      if(i==2){
        fa=c(1.5,1,3,6,12)#fa=c(2,1,4,6,8)
      }
      if(i==3){
        fa=c(3,1.5,1,3,6)#fa=c(3,2,1,4,6)
      }
      if(i==4){
        fa=c(6,3,1.5,1,3)#fa=c(4,3,2,1,4)
      }
      if(i==5){
        fa=c(12,6,3,1.5,1)#fa=c(5,4,3,2,1)
      } 
    }
    if(nrow(ca)>5){
      stop("More classes present than defined in script - please add!")
    }
    ca[i,]=as.numeric(ca[i,])*fa
  }
  
  # Standardize each expected class to 100%:
  
  b1=c()
  for(i in 1:nrow(ca)){
    b1[i]=sum(as.numeric(ca[i,]))
  }
  for(i in 1:nrow(ca)){
    ca[i,]=signif(as.numeric(ca[i,])*100/b1[i],2)
  }
  
  # Recalculate overall accuracy:
  
  pecocl$Accuracy_standardized=NA
  if(nrow(ca)==4){
    a1=c(ca[1,1],ca[2,2],ca[3,3],ca[4,4])
  }
  if(nrow(ca)==5){
    a1=c(ca[1,1],ca[2,2],ca[3,3],ca[4,4],ca[5,5])
  }
  a2=as.numeric(b$n)
  pecocl$Accuracy_standardized=signif(sum(as.numeric(a1)/100*a2)/sum(a2)*100,2)
  
  # Add judgement on how to report:
  
  pecocl$how_to_report=NA
  pecocl$value_1=NA
  pecocl$value_2=NA
  pecocl$value_3=NA
  pecocl[,c("value_1","value_2","value_3")]=boundaries
  if(is.na(pecocl$Accuracy_standardized)!=T){
    if(pecocl$Accuracy>=60 & ((pecocl$Accuracy-pecocl$Accuracy_standardized)<=20 | pecocl$Accuracy_standardized>50)){
      pecocl$how_to_report=c("classes")
    }
    if(pecocl$Accuracy>=80 & ((pecocl$Accuracy-pecocl$Accuracy_standardized)<=12 | pecocl$Accuracy_standardized>70)){
      pecocl$how_to_report=c("numeric")
    }
    if(pecocl$Accuracy<60 | (pecocl$Accuracy-pecocl$Accuracy_standardized)>20){
      pecocl$how_to_report=c("not_report")
    } 
  }
  
  
  
  ## create output and store if asked for:
  
  output<-list(element=element,summaryFertilization=b,extendedFertilization=ext,quartileRMSECV=rms,WEPAL=wep,Accuracy=pecocl,extendedFertilizationStandardized=ca)
  class(output)<-c("pcc")
  
  # Write output to file:
  
  if(sav==TRUE){
    if(exists("loc")==F){
      warning("loc-object is not defined. No plot will be stored on local drive.")
    }
    if(exists("loc")==T){
      if(class(loc)!="character"){
        warning("Class of loc-object has to be 'character'. No plot will be stored on local drive. A plot is generated only.")
      }
      if(class(loc)=="character"){
        if(file.access(loc)==(-1)){
          warning("loc-object does not represent a valid location on local harddrive. No plot will be stored on local drive. A plot is generated only.")
        }
        if(file.access(loc)==0){
          setwd(loc)
          
          # Write files:
          
          if(is.na(nam)==T){
            nam<-output$element
          }
          write.csv(output$summaryFertilization,file=paste(nam," summary classification.csv",sep=""))
          write.csv(output$extendedFertilization,file=paste(nam," extended classification.csv",sep=""))
          write.csv(output$quartileRMSECV,file=paste(nam," quartile RMSECV.csv",sep=""))
          write.csv(output$WEPAL,file=paste(nam," quartile Wepal relation.csv",sep=""))
          
          # Write workspace:
          
          save(output,file=paste(nam,".RData",sep=""))
          
        }
      }
    }
  }
  
  # Give output back to function:
  
  return(output)
  
}


chemicals <- c('boron', 'calcium', 'clay', 'copper', 'ec_salts',
               'exchangeable_acidity', 'iron', 'magnesium', 'manganese', 'phosphorus',
               'potassium', 'sand', 'silt', 'sodium', 'sulphur', 'zinc', 'ph')

for (chem in chemicals) {
  f <- paste("D://CropNutsDocuments/DSML124/outputFiles/PCC_Classes/",chem,".csv")
  df <- as.data.frame(read.csv(file = f)))
  bdr <- c(0.5,0.8,1.0)
  pcc<-function(chem,bdr,df,1,2)
}