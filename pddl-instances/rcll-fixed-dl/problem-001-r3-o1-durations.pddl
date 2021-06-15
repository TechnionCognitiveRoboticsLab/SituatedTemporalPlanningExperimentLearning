(define (problem rcll-production-001-durative)
	(:domain rcll-production-durative)
	(:objects R-1 - robot R-2 - robot R-3 - robot o1 - order wp1 - workpiece cg1 - cap-carrier cg2 - cap-carrier cg3 - cap-carrier cb1 - cap-carrier cb2 - cap-carrier cb3 - cap-carrier C-BS - mps C-CS1 - mps C-CS2 - mps C-DS - mps CYAN - team-color)
	(:init 
		(order-delivery-window-open o1)
		(at 150 (not (order-delivery-window-open o1)))
		(can-commit-for-ontime-delivery o1)
		(at 15 (not (can-commit-for-ontime-delivery o1)))
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
		(order-base-color o1 BASE_RED)
		(order-cap-color o1 CAP_BLACK)
		(order-gate o1 GATE-2)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.706504)
		(= (path-length C-BS INPUT C-CS1 INPUT) 10.380778)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 12.061826)
		(= (path-length C-BS INPUT C-CS2 INPUT) 6.06188)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 5.552087)
		(= (path-length C-BS INPUT C-DS INPUT) 5.538318)
		(= (path-length C-BS INPUT C-DS OUTPUT) 7.619994)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.706504)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 10.484131)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 12.165178)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 6.962768)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 5.655439)
		(= (path-length C-BS OUTPUT C-DS INPUT) 5.64167)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 7.723346)
		(= (path-length C-CS1 INPUT C-BS INPUT) 10.380776)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 10.484129)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 2.808715)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 10.169497)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 9.108337)
		(= (path-length C-CS1 INPUT C-DS INPUT) 9.363879)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 6.437867)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 12.061825)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 12.165178)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 2.808715)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 9.699714)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 8.638554)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 9.076654)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.968083)
		(= (path-length C-CS2 INPUT C-BS INPUT) 6.061879)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 6.962769)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 10.169498)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 9.699715)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.369282)
		(= (path-length C-CS2 INPUT C-DS INPUT) 3.807383)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 3.857314)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 5.552087)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 5.655439)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 9.108338)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 8.638555)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.369282)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 1.166348)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 2.796154)
		(= (path-length C-DS INPUT C-BS INPUT) 5.538318)
		(= (path-length C-DS INPUT C-BS OUTPUT) 5.64167)
		(= (path-length C-DS INPUT C-CS1 INPUT) 9.363881)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 9.076654)
		(= (path-length C-DS INPUT C-CS2 INPUT) 3.807383)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 1.166348)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.234255)
		(= (path-length C-DS OUTPUT C-BS INPUT) 7.619995)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 7.723347)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 6.437868)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.968084)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 3.857314)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 2.796154)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.234255)
		(= (path-length START INPUT C-BS INPUT) 2.830853)
		(= (path-length START INPUT C-BS OUTPUT) 1.955897)
		(= (path-length START INPUT C-CS1 INPUT) 9.436586)
		(= (path-length START INPUT C-CS1 OUTPUT) 11.117634)
		(= (path-length START INPUT C-CS2 INPUT) 5.915224)
		(= (path-length START INPUT C-CS2 OUTPUT) 4.607895)
		(= (path-length START INPUT C-DS INPUT) 4.594126)
		(= (path-length START INPUT C-DS OUTPUT) 6.675802)
	)
	(:goal (order-fulfilled o1))
)