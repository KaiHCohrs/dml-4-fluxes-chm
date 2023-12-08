install.packages('mlegp')
install.packages("REddyProc")
library(REddyProc)
library(dplyr)
library(mlegp)
library(sys)

#site = 'AU-Dap'
#version = 'simple'
#Q10 = '1.5'

site_meta <- read.csv('~/Repositories/data/fluxnet/site_meta.csv', header=TRUE)
rownames(site_meta) <- site_meta$X

files <- list.files('~/Repositories/data/generated_data/')

for (file in files)
{
  if (grepl("part", file, fixed=TRUE) || paste(substr(file, 1, nchar(file)-4), "_part.csv", sep='') %in% files) 
  {
    print(file)
    print("done already")
    next
  }
  start_time <- Sys.time()
  EddyData <- read.csv(paste('~/Repositories/data/generated_data/',file, sep=""))
  for (nee in colnames(EddyData)[grepl( "NEE" , names(EddyData))])
  {
    EddyData <- rename(EddyData, NEE = nee)
    EddyDataWithPosix <- fConvertTimeToPosix(
    EddyData, 'YDH',Year = 'Year',Day = 'DoY', Hour = 'Hour') %>% 
    filterLongRuns("NEE")
    
    #+++ Initalize R5 reference class sEddyProc for post-processing of eddy data
    #+++ with the variables needed for post-processing later
    site = strsplit(file, split="_")[[1]][1]
    EProc <- sEddyProc$new(
      site, EddyDataWithPosix, c('NEE', 'Rg', 'Tair', 'VPD', 'Ustar'))    #
    EProc$sSetLocationInfo(LatDeg = site_meta[site,][2][[1]], LongDeg =site_meta[site,][3][[1]], TimeZoneHour = site_meta[site,][4][[1]])
    EProc$sMDSGapFill('Tair', FillAll = FALSE,  minNWarnRunLength = NA)     
    EProc$sMDSGapFill('VPD', FillAll = FALSE,  minNWarnRunLength = NA)     
    EProc$sMDSGapFill('Rg', FillAll = FALSE,  minNWarnRunLength = NA)
    EProc$sMDSGapFill('NEE', FillAll = FALSE,  minNWarnRunLength = NA)
    
    EProc$sMRFluxPartition()
    EddyData$RECO_NT <- EProc$sTEMP$Reco
    EddyData$GPP_NT <- EProc$sTEMP$GPP_f
      
    EProc$sGLFluxPartition()
    EddyData$RECO_DT <- EProc$sTEMP$Reco_DT
    EddyData$GPP_DT <- EProc$sTEMP$GPP_DT
    
    gpp_dt <- paste("GPP", substr(nee, 4, nchar(nee)), "_DT", sep="")
    reco_dt <- paste("RECO", substr(nee, 4, nchar(nee)), "_DT", sep="")
    gpp_nt <- paste("GPP", substr(nee, 4, nchar(nee)), "_NT", sep="")
    reco_nt <- paste("RECO", substr(nee, 4, nchar(nee)), "_NT", sep="")
    
    EddyData <- rename(EddyData, 
                       !!nee := NEE, 
                       !!gpp_dt := GPP_DT,
                       !!reco_dt := RECO_DT,
                       !!gpp_nt := GPP_NT,
                       !!reco_nt  := RECO_NT)
  }
  
  write.csv(EddyData, paste('~/Repositories/data/generated_data/', substr(file, 1, nchar(file)-4), "_part.csv", sep=""))
  end_time <- Sys.time()
  end_time - start_time
}