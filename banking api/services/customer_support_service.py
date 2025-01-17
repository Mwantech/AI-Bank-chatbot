from models.models import db, CustomerSupport

class CustomerSupportService:
    @staticmethod
    def create_support_request(user_id, request_type, details):
        try:
            support_request = CustomerSupport(
                UserID=user_id,
                RequestType=request_type,
                RequestDetails=details
            )
            db.session.add(support_request)
            db.session.commit()
            return True, "Support request created successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)

    @staticmethod
    def get_support_requests(user_id):
        requests = CustomerSupport.query.filter_by(UserID=user_id).all()
        return [{
            'request_type': req.RequestType,
            'details': req.RequestDetails,
            'status': req.Status,
            'created_at': req.CreatedAt
        } for req in requests]