(define (problem rcll-production-052-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    R-2 - robot
    R-3 - robot
    o1 - order
    wp1 - workpiece
    cg1 cg2 cg3 cb1 cb2 cb3 - cap-carrier
    C-BS C-CS1 C-CS2 C-DS - mps
    CYAN - team-color
  )
   
  (:init
   (mps-type C-BS BS)
   (mps-type C-CS1 CS)
   (mps-type C-CS2 CS) 
   (mps-type C-DS DS)
   (location-free START INPUT)
   (location-free C-BS INPUT)
   (location-free C-BS OUTPUT)
   (location-free C-CS1 INPUT)
   (location-free C-CS1 OUTPUT)
   (location-free C-CS2 INPUT)
   (location-free C-CS2 OUTPUT)
   (location-free C-DS INPUT)
   (location-free C-DS OUTPUT)
   (cs-can-perform C-CS1 CS_RETRIEVE)
   (cs-can-perform C-CS2 CS_RETRIEVE)
   (cs-free C-CS1)
   (cs-free C-CS2)

   (wp-base-color wp1 BASE_NONE)
   (wp-cap-color wp1 CAP_NONE)
   (wp-ring1-color wp1 RING_NONE)
   (wp-ring2-color wp1 RING_NONE)
   (wp-ring3-color wp1 RING_NONE)
   (wp-unused wp1)
   (robot-waiting R-1)
   (robot-waiting R-2)
   (robot-waiting R-3)

   (mps-state C-BS IDLE)
   (mps-state C-CS1 IDLE)
   (mps-state C-CS2 IDLE)
   (mps-state C-DS IDLE)

   (wp-cap-color cg1 CAP_GREY)
   (wp-cap-color cg2 CAP_GREY)
   (wp-cap-color cg3 CAP_GREY)
   (wp-on-shelf cg1 C-CS1 LEFT)
   (wp-on-shelf cg2 C-CS1 MIDDLE)
   (wp-on-shelf cg3 C-CS1 RIGHT)

   (wp-cap-color cb1 CAP_BLACK)
   (wp-cap-color cb2 CAP_BLACK)
   (wp-cap-color cb3 CAP_BLACK)
   (wp-on-shelf cb1 C-CS2 LEFT)
   (wp-on-shelf cb2 C-CS2 MIDDLE)
   (wp-on-shelf cb3 C-CS2 RIGHT)
   (order-complexity o1 c0)
   (order-base-color o1 BASE_BLACK)
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-2)



   (= (path-length C-BS INPUT C-BS OUTPUT) 3.658432)
   (= (path-length C-BS INPUT C-CS1 INPUT) 6.452790)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 7.388668)
   (= (path-length C-BS INPUT C-CS2 INPUT) 13.667389)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 10.482670)
   (= (path-length C-BS INPUT C-DS INPUT) 7.668040)
   (= (path-length C-BS INPUT C-DS OUTPUT) 9.284699)
   (= (path-length C-BS OUTPUT C-BS INPUT) 3.658432)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 4.248331)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 4.689743)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 11.577868)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 7.783746)
   (= (path-length C-BS OUTPUT C-DS INPUT) 6.066575)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 6.585775)
   (= (path-length C-CS1 INPUT C-BS INPUT) 6.452791)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 4.248331)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.442228)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 7.779920)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 5.732728)
   (= (path-length C-CS1 INPUT C-DS INPUT) 5.819059)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 4.534757)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 7.388668)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 4.689743)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.442228)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 8.145940)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.186167)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 3.677275)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 2.988197)
   (= (path-length C-CS2 INPUT C-BS INPUT) 13.667388)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 11.577868)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 7.779919)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 8.145940)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 5.469701)
   (= (path-length C-CS2 INPUT C-DS INPUT) 9.226685)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 7.551854)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 10.482669)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 7.783745)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 5.732727)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.186167)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 5.469701)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 6.541922)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 4.053337)
   (= (path-length C-DS INPUT C-BS INPUT) 7.668039)
   (= (path-length C-DS INPUT C-BS OUTPUT) 6.066574)
   (= (path-length C-DS INPUT C-CS1 INPUT) 5.819058)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 3.677275)
   (= (path-length C-DS INPUT C-CS2 INPUT) 9.226686)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 6.541922)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.279485)
   (= (path-length C-DS OUTPUT C-BS INPUT) 9.284698)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 6.585774)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 4.534756)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 2.988197)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 7.551853)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 4.053337)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.279484)
   (= (path-length START INPUT C-BS INPUT) 2.962804)
   (= (path-length START INPUT C-BS OUTPUT) 1.518303)
   (= (path-length START INPUT C-CS1 INPUT) 3.568104)
   (= (path-length START INPUT C-CS1 OUTPUT) 5.261355)
   (= (path-length START INPUT C-CS2 INPUT) 10.782701)
   (= (path-length START INPUT C-CS2 OUTPUT) 8.355358)
   (= (path-length START INPUT C-DS INPUT) 6.638186)
   (= (path-length START INPUT C-DS OUTPUT) 7.157388))

  (:goal (order-fulfilled o1))
)
