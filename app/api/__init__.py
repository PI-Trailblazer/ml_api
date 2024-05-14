from fastapi import APIRouter

from . import ml

router = APIRouter()

router.include_router(ml.router, prefix="/ml", tags=["ml"])
