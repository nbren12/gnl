; ===============================================
; h_time_1.ncl              
; ===============================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin
;================================================
; get data
;================================================
 fili  =  "toga_rg.nc"                                 ; filename
 
 f     = addfile (fili , "r")                     ; add file 

 v2D = f->u
 v2Dt = transpose(v2D)
;================================================
; default plot
;================================================
 wks                 = gsn_open_wks ("pdf", "h_time" )     ; open workstation
 res                 = True                               ; plot mods desired
 res@tiMainString    = "Zonal Mean at 1.4S"               ; title
 res@cnLevelSpacingF = 5                                ; contour spacing

; note this data is already on pressure levels. If this were model
; data, it would be necessary to convert from hybrid to pressure levels
; before plotting.

 plot = gsn_csm_pres_hgt (wks,v2Dt,res) 
end
