from app.models.alert import Alert
from datetime import datetime
def check_and_generate_alerts(db, zone_id, people, vehicles, density, risk):
    if risk in ["HIGH","CRITICAL"]:
        db.add(Alert(alert_type="HIGH_DENSITY" if risk=="HIGH" else "CRITICAL_DENSITY", zone_id=zone_id, risk_level=risk, message=f"High density in {zone_id}: {density:.1f}%", created_at=datetime.utcnow()))
    if zone_id=="SENSITIVE_ZONE" and people>0:
        db.add(Alert(alert_type="RESTRICTED_ZONE_ENTRY", zone_id=zone_id, risk_level=risk, message="Person entered SENSITIVE_ZONE", created_at=datetime.utcnow()))
