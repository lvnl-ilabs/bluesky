from bluesky.traffic.performance.perfbase import PerfBase

performance_model = PerfBase.selected().__name__.lower()

if performance_model == "openap":
    PHASE = {
        "None":0,
        "GD": 1,    # Take-off
        "IC": 2,    # Initial Climb
        "CL": 3,    # Climb
        "CR": 4,    # Cruise
        "DE": 5,    # Descent
        "AP": 6,    # Approach
        "gd": 1,    # and lower case to be sure
        "ic": 2,
        "cl": 3,
        "cr": 4,
        "de": 5,
        "ap": 6
    }

else:
    PHASE = {"None":0,
             "TO"  :1,      # Take-off
             "IC"  :2,      # Initial climb
             "CL"  :3,      # Climb
             "CR"  :4,      # Cruise
             "SC"  :41,     # Step Climb (Cruise)
             "SD"  :42,     # Step Descent (Cruise)
             "DE"  :5,      # Descent
             "AP"  :6,      # Approach
             "LD"  :7,      # Landing
             "GD"  :8,      # Ground
             "to"  :1,
             "ic"  :2,
             "cl"  :3,
             "cr"  :4,
             "sc"  :41,
             "sd"  :42,
             "de"  :5,      # and lower case to be sure
             "ap"  :6,
             "ld"  :7,
             "gd"  :8,
            }