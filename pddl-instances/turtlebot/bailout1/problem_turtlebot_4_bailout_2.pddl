(define (problem task)
(:domain turtlebot)
(:objects
    entrance dock-station meeting-room dan-office phdarea fridge coffee - waypoint
    printer-ent printer-corridor printer-phdarea - printer
    kenny - robot
)
(:init

    (undocked kenny)
	(localised kenny)

    (dock_at dock-station)
    (nocarrying_papers kenny)

    (delivery_destination phdarea)
	(at 957.635 (not (delivery_destination phdarea)))

	(at 157.635 (delivery_destination meeting-room))
	(at 757.635 (not (delivery_destination meeting-room)))

    (delivery_destination coffee)
	(at 657.635 (not (delivery_destination coffee)))

    (delivery_destination dan-office)
	(at 157.635 (not (delivery_destination dan-office)))

	(bailout_available)	
	(bailout_location dock-station)	
	(= (bailout_distance) 9.48)
	(at 17 (= (bailout_distance) 9.49))
	(at 34 (= (bailout_distance) 12.49))
	(at 51 (= (bailout_distance) 15.49))
	(at 68 (= (bailout_distance) 18.77))
	(at 72 (= (bailout_distance) 19.69))

	(= (distance entrance dock-station) 6.865)
	(= (distance entrance meeting-room) 5.399)
	(= (distance entrance dan-office) 6.871)
	(= (distance entrance printer-ent) 2.248)
	(= (distance entrance printer-corridor) 23.376)
	(= (distance entrance printer-phdarea) 23.383)
	(= (distance entrance phdarea) 27.089)
	(= (distance entrance fridge) 10.219)
	(= (distance entrance coffee) 3.281)
	(= (distance dock-station entrance) 6.865)
	(= (distance dock-station meeting-room) 10.521)
	(= (distance dock-station dan-office) 13.736)
	(= (distance dock-station printer-ent) 7.225)
	(= (distance dock-station printer-corridor) 19.685)
	(= (distance dock-station printer-phdarea) 19.692)
	(= (distance dock-station phdarea) 23.398)
	(= (distance dock-station fridge) 3.354)
	(= (distance dock-station coffee) 3.584)
	(= (distance meeting-room entrance) 5.399)
	(= (distance meeting-room dock-station) 10.521)
	(= (distance meeting-room dan-office) 3.216)
	(= (distance meeting-room printer-ent) 3.295)
	(= (distance meeting-room printer-corridor) 17.977)
	(= (distance meeting-room printer-phdarea) 17.985)
	(= (distance meeting-room phdarea) 21.69)
	(= (distance meeting-room fridge) 13.875)
	(= (distance meeting-room coffee) 6.937)
	(= (distance dan-office entrance) 6.871)
	(= (distance dan-office dock-station) 13.736)
	(= (distance dan-office meeting-room) 3.216)
	(= (distance dan-office printer-ent) 6.511)
	(= (distance dan-office printer-corridor) 17.218)
	(= (distance dan-office printer-phdarea) 17.349)
	(= (distance dan-office phdarea) 20.931)
	(= (distance dan-office fridge) 17.091)
	(= (distance dan-office coffee) 10.152)
	(= (distance printer-ent entrance) 2.248)
	(= (distance printer-ent dock-station) 7.225)
	(= (distance printer-ent meeting-room) 3.295)
	(= (distance printer-ent dan-office) 6.511)
	(= (distance printer-ent printer-corridor) 21.128)
	(= (distance printer-ent printer-phdarea) 21.135)
	(= (distance printer-ent phdarea) 24.841)
	(= (distance printer-ent fridge) 10.58)
	(= (distance printer-ent coffee) 3.641)
	(= (distance printer-corridor entrance) 23.376)
	(= (distance printer-corridor dock-station) 19.685)
	(= (distance printer-corridor meeting-room) 17.977)
	(= (distance printer-corridor dan-office) 17.218)
	(= (distance printer-corridor printer-ent) 21.128)
	(= (distance printer-corridor printer-phdarea) 12.364)
	(= (distance printer-corridor phdarea) 5.684)
	(= (distance printer-corridor fridge) 19.762)
	(= (distance printer-corridor coffee) 23.032)
	(= (distance printer-phdarea entrance) 23.383)
	(= (distance printer-phdarea dock-station) 19.692)
	(= (distance printer-phdarea meeting-room) 17.985)
	(= (distance printer-phdarea dan-office) 17.349)
	(= (distance printer-phdarea printer-ent) 21.135)
	(= (distance printer-phdarea printer-corridor) 12.364)
	(= (distance printer-phdarea phdarea) 6.679)
	(= (distance printer-phdarea fridge) 19.769)
	(= (distance printer-phdarea coffee) 23.04)
	(= (distance phdarea entrance) 27.089)
	(= (distance phdarea dock-station) 23.398)
	(= (distance phdarea meeting-room) 21.69)
	(= (distance phdarea dan-office) 20.931)
	(= (distance phdarea printer-ent) 24.841)
	(= (distance phdarea printer-corridor) 5.684)
	(= (distance phdarea printer-phdarea) 6.679)
	(= (distance phdarea fridge) 23.474)
	(= (distance phdarea coffee) 26.745)
	(= (distance fridge entrance) 10.219)
	(= (distance fridge dock-station) 3.354)
	(= (distance fridge meeting-room) 13.875)
	(= (distance fridge dan-office) 17.091)
	(= (distance fridge printer-ent) 10.58)
	(= (distance fridge printer-corridor) 19.762)
	(= (distance fridge printer-phdarea) 19.769)
	(= (distance fridge phdarea) 23.474)
	(= (distance fridge coffee) 6.938)
	(= (distance coffee entrance) 3.281)
	(= (distance coffee dock-station) 3.584)
	(= (distance coffee meeting-room) 6.937)
	(= (distance coffee dan-office) 10.152)
	(= (distance coffee printer-ent) 3.641)
	(= (distance coffee printer-corridor) 23.032)
	(= (distance coffee printer-phdarea) 23.04)
	(= (distance coffee phdarea) 26.745)
	(= (distance coffee fridge) 6.938)
)
(:goal (and
    ;; (papers_delivered kenny dan-office)
    (papers_delivered kenny phdarea)
    (papers_delivered kenny coffee)
    (papers_delivered kenny meeting-room)
    (docked kenny)
    )
)
)
