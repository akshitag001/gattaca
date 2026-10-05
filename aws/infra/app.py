#!/usr/bin/env python3
"""AWS CDK app for GATTACA.

    cd aws/infra && pip install -r requirements.txt
    cdk deploy -c coolAmiId=ami-XXXXXXXX -c artifactBucket=<bucket> [-c withX86Baseline=true]

`coolAmiId` is the AMI of "Cloud Optimized OpenCV For AWS Graviton4" (AWS Marketplace,
v3.1, OpenCV 5.0). Subscribe in the Marketplace console first, then copy the
AMI id for your region from the listing's launch page.
"""

import aws_cdk as cdk

from gattaca_stack import GattacaStack

app = cdk.App()
GattacaStack(
    app,
    "Gattaca",
    env=cdk.Environment(account=app.node.try_get_context("account"),
                        region=app.node.try_get_context("region") or "us-east-1"),
)
app.synth()
